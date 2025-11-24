"""Command-line utility to download Yahoo Finance OHLCV data into CSV files.

The tool is designed for long-running historical pulls, repeatable incremental
updates, and lightweight automation use. It supports multiple symbols,
optional adjusted close data, and JSON metadata summaries to assist data
pipelines.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from numbers import Number
from typing import Iterable, Iterator, Optional, Sequence

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - dependency bootstrap
    print("yfinance library not found. Installing...", flush=True)
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf  # type: ignore  # noqa: E402


DEFAULT_SYMBOL = "MASTEK.NS"
DEFAULT_INTERVAL = "1d"
DEFAULT_OUTPUT = "MASTEK_OHLCV.csv"
SUPPORTED_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


@dataclass(slots=True)
class Config:
    symbols: tuple[str, ...]
    start: Optional[dt.date]
    end: Optional[dt.date]
    interval: str
    auto_adjust: bool
    output: str
    split_output: bool
    quiet: bool
    metadata_path: Optional[str]
    force: bool
    incremental: bool


def parse_args(argv: Optional[Iterable[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from Yahoo Finance and export to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        dest="single_symbols",
        action="append",
        help="Yahoo Finance ticker symbol (can be provided multiple times).",
    )
    parser.add_argument(
        "--symbols",
        dest="comma_symbols",
        help="Comma-separated list of ticker symbols (alternative to --symbol).",
    )
    parser.add_argument(
        "--symbols-file",
        dest="symbols_file",
        help="Path to a text file with one ticker symbol per line.",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Start date in YYYY-MM-DD format. Defaults to earliest available.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="End date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=sorted(SUPPORTED_INTERVALS),
        help="Candle interval to download.",
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Return adjusted OHLC values (default uses raw close).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV path. When --split-output is set, treated as a directory "
            "where files are emitted as <symbol>.csv"
        ),
    )
    parser.add_argument(
        "--split-output",
        action="store_true",
        help="Write one CSV per symbol instead of a combined file.",
    )
    parser.add_argument(
        "--metadata",
        dest="metadata_path",
        help="Optional path to JSON file with download metadata summary.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (errors still print to stderr).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files without prompting.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Append only new candles to existing CSV output (auto-detects last saved date).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    resolved_symbols = _resolve_symbols(
        cli_symbols=args.single_symbols,
        comma_symbols=args.comma_symbols,
        symbols_file=args.symbols_file,
    )

    if not resolved_symbols:
        resolved_symbols = (DEFAULT_SYMBOL,)

    return Config(
        symbols=resolved_symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=args.auto_adjust,
        output=args.output,
        split_output=args.split_output,
        quiet=args.quiet,
        metadata_path=args.metadata_path,
        force=args.force,
        incremental=args.incremental,
    )


def _resolve_symbols(
    *,
    cli_symbols: Optional[Sequence[str]],
    comma_symbols: Optional[str],
    symbols_file: Optional[str],
) -> tuple[str, ...]:
    symbols: list[str] = []

    if cli_symbols:
        for value in cli_symbols:
            symbols.extend(_split_symbol_list(value))

    if comma_symbols:
        symbols.extend(_split_symbol_list(comma_symbols))

    if symbols_file:
        try:
            with open(symbols_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        symbols.append(line)
        except OSError as exc:
            raise SystemExit(f"Unable to read symbols file '{symbols_file}': {exc}")

    # deduplicate while preserving order
    seen: set[str] = set()
    unique_symbols = []
    for symbol in symbols:
        symbol = symbol.strip()
        if not symbol:
            continue
        if symbol.upper() not in seen:
            seen.add(symbol.upper())
            unique_symbols.append(symbol)

    return tuple(unique_symbols)


def _split_symbol_list(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format YYYY-MM-DD"
        ) from exc


def resolve_date_range(start: Optional[dt.date], end: Optional[dt.date]) -> tuple[
    Optional[dt.datetime], Optional[dt.datetime]
]:
    """Convert naive dates to timezone-aware datetimes suitable for yfinance."""

    tz = dt.timezone.utc
    start_dt = dt.datetime.combine(start, dt.time.min, tzinfo=tz) if start else None
    # yfinance treats end as exclusive, so include entire day when provided.
    end_dt = (
        dt.datetime.combine(end + dt.timedelta(days=1), dt.time.min, tzinfo=tz)
        if end
        else None
    )
    return start_dt, end_dt


def download_symbol(
    symbol: str,
    *,
    interval: str,
    start: Optional[dt.datetime],
    end: Optional[dt.datetime],
    auto_adjust: bool,
    quiet: bool,
):
    if not quiet:
        logging.info(
            "Fetching %s data from %s to %s at interval %s",
            symbol,
            start.date() if start else "earliest",
            (end - dt.timedelta(days=1)).date() if end else "today",
            interval,
        )

    download_kwargs = {
        "interval": interval,
        "auto_adjust": auto_adjust,
        "progress": False,
        "actions": False,
        "threads": False,
    }
    if start is not None:
        download_kwargs["start"] = start
    if end is not None:
        download_kwargs["end"] = end
    if start is None and end is None:
        download_kwargs["period"] = "max"

    df = yf.download(symbol, **download_kwargs)

    if df.empty:
        raise RuntimeError(
            f"No data returned for symbol '{symbol}'. Check the ticker or date range."
        )

    df.sort_index(inplace=True)
    df["Symbol"] = symbol
    return df


def _safe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, Number):
        float_value = float(value)
        if math.isnan(float_value):
            return None
        return float_value
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(float_value):
        return None
    return float_value


def _normalise_dataframe(df, symbol: str) -> pd.DataFrame:
    # When downloading multiple symbols, yfinance returns a multi-index column
    # structure. We need to select the data for the correct symbol.
    if isinstance(df.columns, pd.MultiIndex):
        # Filter columns for the current symbol. The first level is OHLCV,
        # the second is the symbol.
        df_for_symbol = df.loc[:, (slice(None), symbol)]
        # Now flatten the column index to just the OHLCV labels.
        df_for_symbol.columns = df_for_symbol.columns.get_level_values(0)
    else:
        # If it's a single-symbol download, the columns are not multi-indexed.
        df_for_symbol = df

    df_copy = df_for_symbol.copy()

    # Ensure standard columns exist, filling with NaN if not present.
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in df_copy.columns:
            df_copy[col] = pd.NA

    # Rename columns to lowercase for consistency
    df_copy.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # Add the symbol column
    df_copy["symbol"] = symbol

    # Enforce OHLC relationship rules
    price_cols = ["open", "high", "low", "close"]
    df_copy[price_cols] = df_copy[price_cols].apply(pd.to_numeric, errors='coerce')
    df_copy["high"] = df_copy[["high", "open", "close"]].max(axis=1)
    df_copy["low"] = df_copy[["low", "open", "close"]].min(axis=1)

    # Recalculate percentage changes using pandas for reliability
    df_copy["pct_change"] = df_copy["close"].pct_change(fill_method=None) * 100
    df_copy["adj_pct_change"] = df_copy["adj_close"].pct_change(fill_method=None) * 100

    # Add the date column from the index, handling timezone localization
    date_series = df_copy.index.to_series()
    if date_series.dt.tz is None:
        df_copy["date"] = date_series.dt.tz_localize("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    else:
        df_copy["date"] = date_series.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Clean up volume column
    df_copy["volume"] = pd.to_numeric(df_copy["volume"], errors="coerce").fillna(0).astype(int)

    # Select and reorder columns for the final output
    final_cols = [
        "symbol", "date", "open", "high", "low", "close",
        "adj_close", "volume", "pct_change", "adj_pct_change"
    ]

    return df_copy[[col for col in final_cols if col in df_copy.columns]]


STANDARD_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "pct_change",
    "adj_pct_change",
]


def _build_normalised_dataframe(df, include_symbol: bool, symbol: str) -> pd.DataFrame:
    normalised = _normalise_dataframe(df, symbol)
    if not include_symbol and "symbol" in normalised.columns:
        normalised = normalised.drop(columns=["symbol"])

    desired_columns = list(STANDARD_COLUMNS)
    if include_symbol and "symbol" in normalised.columns:
        desired_columns.insert(0, "symbol")

    existing_columns = [col for col in desired_columns if col in normalised.columns]
    return normalised.reindex(columns=existing_columns)


def _merge_normalised(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    include_symbol: bool,
) -> pd.DataFrame:
    if existing_df.empty:
        return new_df.copy()

    work_existing = existing_df.copy()
    work_new = new_df.copy()

    # Ensure consistent column ordering and dtypes prior to merge.
    for column in STANDARD_COLUMNS:
        if column in work_existing.columns and column in work_new.columns:
            if column == "date":
                work_existing[column] = pd.to_datetime(
                    work_existing[column], utc=True, errors="coerce"
                )
                work_new[column] = pd.to_datetime(
                    work_new[column], utc=True, errors="coerce"
                )
            elif column == "volume":
                work_existing[column] = (
                    pd.to_numeric(work_existing[column], errors="coerce")
                    .fillna(0)
                    .round()
                    .astype(int)
                )
                work_new[column] = (
                    pd.to_numeric(work_new[column], errors="coerce")
                    .fillna(0)
                    .round()
                    .astype(int)
                )
            else:
                work_existing[column] = pd.to_numeric(
                    work_existing[column], errors="coerce"
                )
                work_new[column] = pd.to_numeric(work_new[column], errors="coerce")

    all_columns: list[str] = list(work_existing.columns)
    for column in work_new.columns:
        if column not in all_columns:
            all_columns.append(column)
    work_existing = work_existing.reindex(columns=all_columns)
    work_new = work_new.reindex(columns=all_columns)

    work_existing["_merge_dt"] = pd.to_datetime(work_existing.get("date"), utc=True, errors="coerce")
    work_new["_merge_dt"] = pd.to_datetime(work_new.get("date"), utc=True, errors="coerce")

    combined = pd.concat([work_existing, work_new], ignore_index=True, sort=False)
    combined = combined.dropna(subset=["_merge_dt"])

    subset_cols = ["_merge_dt"]
    if include_symbol and "symbol" in combined.columns:
        subset_cols.insert(0, "symbol")

    combined = combined.drop_duplicates(subset=subset_cols, keep="last")
    combined = combined.sort_values(subset_cols)
    combined = combined.drop(columns=["_merge_dt"])
    combined.reset_index(drop=True, inplace=True)

    if "date" in combined.columns:
        combined["date"] = (
            pd.to_datetime(combined["date"], utc=True, errors="coerce")
            .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        )

    return combined


def _clean_volume(value) -> int:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, Number):
        float_value = float(value)
        if math.isnan(float_value):
            return 0
        return int(round(float_value))
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def write_csv(
    df,
    *,
    output_path: str,
    is_normalized: bool,
    include_symbol_column: bool,
    symbol: str,
    force: bool,
    incremental: bool,
):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if is_normalized:
        normalised_df = df
    else:
        normalised_df = _build_normalised_dataframe(df, include_symbol_column, symbol)

    if incremental and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
        except Exception as exc:
            logging.warning("Unable to read existing output '%s' for incremental merge (%s). Recreating file.", output_path, exc)
            existing_df = pd.DataFrame(columns=normalised_df.columns)
        merged = _merge_normalised(existing_df, normalised_df, include_symbol_column)
        merged.to_csv(output_path, index=False)
        logging.info(
            "Merged %s new rows into %s",
            max(len(merged) - len(existing_df), 0),
            os.path.abspath(output_path),
        )
    else:
        if os.path.exists(output_path) and not force:
            raise FileExistsError(
                f"Output file '{output_path}' already exists. Use --force to overwrite."
            )
        normalised_df.to_csv(output_path, index=False)
        logging.info("Saved %s rows to %s", len(normalised_df), os.path.abspath(output_path))


def _write_metadata(
    *,
    metadata_path: str,
    results: list[dict[str, object]],
    force: bool,
):
    output_dir = os.path.dirname(os.path.abspath(metadata_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(metadata_path) and not force:
        raise FileExistsError(
            f"Metadata file '{metadata_path}' already exists. Use --force to overwrite."
        )

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=_json_default)
        handle.write("\n")

    logging.info("Wrote metadata summary to %s", os.path.abspath(metadata_path))


def _json_default(value):
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _build_metadata(symbol: str, df) -> dict[str, object]:
    index = df.index
    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "start": index[0].isoformat() if len(index) else None,
        "end": index[-1].isoformat() if len(index) else None,
        "interval": df.attrs.get("Interval"),
    }


def _ensure_output_target(path: str, *, split_output: bool):
    if split_output:
        if path.lower().endswith(".csv"):
            logging.warning(
                "Treating output '%s' as directory because --split-output is set.",
                path,
            )
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        elif not os.path.isdir(path):
            raise SystemExit(
                "When --split-output is used the --output argument must be a directory"
            )
    else:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = parse_args(argv)

    logging.basicConfig(
        level=logging.ERROR if cfg.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg.end and cfg.start and cfg.end < cfg.start:
        raise SystemExit("--end date must be on or after --start date")

    _ensure_output_target(cfg.output, split_output=cfg.split_output)

    combined_frames = []
    metadata_entries: list[dict[str, object]] = []
    downloaded_anything = False

    for symbol in cfg.symbols:
        symbol_start = cfg.start
        symbol_end = cfg.end

        if cfg.split_output:
            target_path = os.path.join(cfg.output, f"{symbol.replace(':', '_')}.csv")
        else:
            target_path = cfg.output

        if cfg.force and os.path.exists(target_path):
            logging.info("Force flag set. Deleting existing file: %s", target_path)
            try:
                os.remove(target_path)
            except OSError as e:
                logging.error("Failed to delete %s: %s", target_path, e)
                continue # Skip to the next symbol

        if cfg.incremental and not cfg.force:
            last_saved: Optional[dt.date] = None
            if target_path and os.path.exists(target_path):
                try:
                    existing_df = pd.read_csv(target_path)
                except Exception as exc:
                    logging.warning(
                        "Unable to read existing data for %s (%s). Skipping incremental optimisation.",
                        symbol,
                        exc,
                    )
                else:
                    if not existing_df.empty and "date" in existing_df.columns:
                        candidate_df = existing_df
                        if not cfg.split_output and "symbol" in existing_df.columns:
                            candidate_df = candidate_df[candidate_df["symbol"].str.upper() == symbol.upper()]

                        if not candidate_df.empty:
                            last_ts = pd.to_datetime(
                                candidate_df["date"], utc=True, errors="coerce"
                            ).dropna().max()
                            if pd.notna(last_ts):
                                last_saved = last_ts.date()

            if last_saved:
                next_day = last_saved + dt.timedelta(days=1)
                if symbol_start is None or next_day > symbol_start:
                    symbol_start = next_day
                effective_end = symbol_end or dt.datetime.now(dt.timezone.utc).date()
                if next_day > effective_end:
                    logging.info("%s already up to date through %s", symbol, last_saved)
                    continue

        start_dt, end_dt = resolve_date_range(symbol_start, symbol_end)

        try:
            df = download_symbol(
                symbol,
                interval=cfg.interval,
                start=start_dt,
                end=end_dt,
                auto_adjust=cfg.auto_adjust,
                quiet=cfg.quiet,
            )
        except Exception as exc:
            logging.error("Failed to download %s: %s", symbol, exc)
            continue

        df.attrs["Interval"] = cfg.interval
        metadata_entries.append(_build_metadata(symbol, df))
        downloaded_anything = True

        if cfg.split_output:
            output_file = os.path.join(cfg.output, f"{symbol.replace(':', '_')}.csv")
            write_csv(
                df,
                output_path=output_file,
                is_normalized=False,
                include_symbol_column=False,
                symbol=symbol,
                force=cfg.force,
                incremental=cfg.incremental,
            )
        else:
            combined_frames.append(df)

    if not cfg.split_output and combined_frames:
        merged = pd.concat(combined_frames)
        merged.sort_index(inplace=True)
        # For combined files, we need to iterate and normalize per symbol
        # This is a simplified approach; a more robust solution would group by symbol
        # and normalize each group.
        if len(cfg.symbols) > 1:
            all_normalised = []
            # When downloading multiple symbols together, yfinance returns a
            # multi-index dataframe that needs careful handling.
            # The download_symbol function adds a 'Symbol' column, but the main
            # yf.download call returns a wide dataframe.
            # We will process the combined dataframe which has a simple structure.
            merged_df = pd.concat(combined_frames)
            for sym in merged_df['Symbol'].unique():
                sym_df = merged_df[merged_df['Symbol'] == sym]
                # The dataframe passed to _build_normalised_dataframe should not have multi-level columns
                # It expects the raw single-symbol format.
                all_normalised.append(_build_normalised_dataframe(sym_df, True, sym))
            final_df = pd.concat(all_normalised, ignore_index=True)
            final_df.sort_values(by=['date', 'symbol'], inplace=True)

        else:
            # Single symbol, no splitting
            merged = pd.concat(combined_frames)
            final_df = _build_normalised_dataframe(merged, False, cfg.symbols[0])

        write_csv(
            final_df,
            output_path=cfg.output,
            is_normalized=True,
            include_symbol_column=len(cfg.symbols) > 1,
            symbol=cfg.symbols[0] if len(cfg.symbols) == 1 else "", # Symbol not used when is_normalized=True
            force=cfg.force,
            incremental=cfg.incremental,
        )

    if cfg.metadata_path and metadata_entries:
        _write_metadata(
            metadata_path=cfg.metadata_path,
            results=metadata_entries,
            force=cfg.force or cfg.incremental,
        )

    if not metadata_entries:
        if cfg.incremental and not downloaded_anything:
            logging.info("All symbols already up to date; no new records downloaded.")
            return 0
        logging.error("No data downloaded. See logs for errors.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

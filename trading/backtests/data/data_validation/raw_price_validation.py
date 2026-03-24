import pandas as pd
from pathlib import Path

YEAR = 2024

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
RAW_DIR = DATA_DIR / "raw_data"

INPUT_FILE = RAW_DIR / f"raw_price_data_{YEAR}.csv"
INCONSISTENCY_FILE = BASE_DIR / f"raw_price_data_{YEAR}_inconsistencies.csv"


def main():
    # Load the raw price file from ../raw_data and parse timestamps as UTC-aware datetimes
    df = pd.read_csv(INPUT_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Basic file-level sanity checks: row count, symbol count, and date range
    print("Rows:", len(df))
    print("Symbols:", df["symbol"].nunique())
    print("Min time:", df["time"].min())
    print("Max time:", df["time"].max())
    print()

    # Check for duplicate bars within the same symbol and timestamp
    dupes = df.duplicated(subset=["symbol", "time"]).sum()
    print("Duplicate symbol+time rows:", dupes)

    # Check that timestamps are in strictly increasing order within each symbol
    bad_order_symbols = []
    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("time")
        if not g["time"].is_monotonic_increasing:
            bad_order_symbols.append(symbol)

    print("Symbols with non-monotonic time order:", len(bad_order_symbols))
    if bad_order_symbols:
        print("Bad order symbols:", bad_order_symbols)
    print()

    # Check OHLC integrity: high should be the max and low should be the min of the bar
    bad_ohlc = df[
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df["open"] <= 0) |
        (df["high"] <= 0) |
        (df["low"] <= 0) |
        (df["close"] <= 0)
    ]
    print("Rows with invalid OHLC structure:", len(bad_ohlc))

    # Check that spread and volume-related fields are non-negative
    bad_nonnegative = df[
        (df["tick_volume"] < 0) |
        (df["spread"] < 0) |
        (df["real_volume"] < 0)
    ]
    print("Rows with negative volume/spread fields:", len(bad_nonnegative))
    print()

    # Create a trading-date column so we can validate daily session structure
    df["date"] = df["time"].dt.date

    # Summarize each symbol-day by number of bars, first bar time, and last bar time
    daily = (
        df.groupby(["date", "symbol"], as_index=False)
        .agg(
            bars=("time", "size"),
            first_bar=("time", "min"),
            last_bar=("time", "max"),
        )
        .sort_values(["date", "symbol"])
    )

    daily["first_bar_str"] = daily["first_bar"].dt.strftime("%H:%M:%S")
    daily["last_bar_str"] = daily["last_bar"].dt.strftime("%H:%M:%S")

    # Check for dates where symbols disagree on bar count or session window
    inconsistent_dates = []
    for date, group in daily.groupby("date"):
        same_bars = group["bars"].nunique() == 1
        same_first = group["first_bar_str"].nunique() == 1
        same_last = group["last_bar_str"].nunique() == 1

        if not (same_bars and same_first and same_last):
            inconsistent_dates.append(group)

    if inconsistent_dates:
        result = pd.concat(inconsistent_dates, ignore_index=True)
        result.to_csv(INCONSISTENCY_FILE, index=False)
        print(f"Saved inconsistent dates to {INCONSISTENCY_FILE}")
        print("Number of inconsistent dates:", result["date"].nunique())
    else:
        print("No cross-symbol daily inconsistencies found.")
    print()

    # Summarize the most common session pattern for each symbol across the year
    symbol_summary = (
        daily.groupby("symbol", as_index=False)
        .agg(
            trading_days=("date", "size"),
            min_bars_day=("bars", "min"),
            max_bars_day=("bars", "max"),
            median_bars_day=("bars", "median"),
            most_common_first_bar=("first_bar_str", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            most_common_last_bar=("last_bar_str", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        )
        .sort_values("symbol")
    )

    print("Per-symbol session summary:")
    print(symbol_summary.to_string(index=False))


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path

YEAR = 2024

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
RAW_DIR = DATA_DIR / "input_raw_data"

INPUT_FILE = RAW_DIR / f"raw_price_data_{YEAR}.csv"
INCONSISTENCY_FILE = BASE_DIR / f"raw_price_data_{YEAR}_inconsistencies.csv"


def main():
    # Load raw file
    df = pd.read_csv(INPUT_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Basic file-level sanity checks
    print("Rows:", len(df))
    print("Symbols:", df["symbol"].nunique())
    print("Min time:", df["time"].min())
    print("Max time:", df["time"].max())
    print()

    # Duplicate symbol-timestamp rows
    dupes = df.duplicated(subset=["symbol", "time"]).sum()
    print("Duplicate symbol+time rows:", dupes)

    # Check original ordering within each symbol
    bad_order_symbols = []
    for symbol, g in df.groupby("symbol"):
        if not g["time"].is_monotonic_increasing:
            bad_order_symbols.append(symbol)

    print("Symbols with non-monotonic time order:", len(bad_order_symbols))
    if bad_order_symbols:
        print("Bad order symbols:", bad_order_symbols)
    print()

    # Missing values in core columns
    core_cols = ["symbol", "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    missing_rows = df[core_cols].isna().any(axis=1).sum()
    print("Rows with missing core fields:", missing_rows)

    # OHLC integrity
    bad_ohlc = df[
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df["open"] <= 0) |
        (df["high"] <= 0) |
        (df["low"] <= 0) |
        (df["close"] <= 0)
    ]
    print("Rows with invalid OHLC structure:", len(bad_ohlc))

    # Non-negative spread / volume fields
    bad_nonnegative = df[
        (df["tick_volume"] < 0) |
        (df["spread"] < 0) |
        (df["real_volume"] < 0)
    ]
    print("Rows with negative volume/spread fields:", len(bad_nonnegative))

    # Check that timestamps lie exactly on a 5-minute grid
    bad_grid = df[
        (df["time"].dt.second != 0) |
        (df["time"].dt.microsecond != 0) |
        (df["time"].dt.minute % 5 != 0)
    ]
    print("Rows off the 5-minute grid:", len(bad_grid))
    print()

    # Trading date
    df["date"] = df["time"].dt.date

    # Daily symbol summary
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

    expected_symbols = set(df["symbol"].unique())
    inconsistent_dates = []

    # Strict cross-symbol timestamp alignment check
    for date, group in df.groupby("date"):
        group = group.sort_values(["symbol", "time"]).copy()

        present_symbols = set(group["symbol"].unique())
        missing_symbols = sorted(expected_symbols - present_symbols)

        symbol_times = {}
        for symbol, sg in group.groupby("symbol"):
            timestamps = tuple(
                sg.sort_values("time")["time"].dt.strftime("%H:%M:%S").tolist()
            )
            symbol_times[symbol] = timestamps

        date_is_inconsistent = False
        mismatching_symbols = []

        # Missing symbol for the date
        if missing_symbols:
            date_is_inconsistent = True

        # Full timestamp sequence alignment
        if symbol_times:
            reference_symbol = sorted(symbol_times.keys())[0]
            reference_times = symbol_times[reference_symbol]

            mismatching_symbols = [
                symbol for symbol, times in symbol_times.items()
                if times != reference_times
            ]

            if mismatching_symbols:
                date_is_inconsistent = True

        if date_is_inconsistent:
            inconsistent_daily = daily[daily["date"] == date].copy()
            inconsistent_daily["timestamp_mismatch"] = False
            inconsistent_daily["missing_symbol_on_date"] = False

            if mismatching_symbols:
                inconsistent_daily.loc[
                    inconsistent_daily["symbol"].isin(mismatching_symbols),
                    "timestamp_mismatch"
                ] = True

            inconsistent_daily["missing_symbols_count"] = len(missing_symbols)
            inconsistent_daily["missing_symbols"] = ", ".join(missing_symbols)

            inconsistent_dates.append(inconsistent_daily)

    if inconsistent_dates:
        result = pd.concat(inconsistent_dates, ignore_index=True)
        result.to_csv(INCONSISTENCY_FILE, index=False)
        print(f"Saved inconsistent dates to {INCONSISTENCY_FILE}")
        print("Number of inconsistent dates:", result["date"].nunique())
        print("Number of symbol-date rows written:", len(result))
    else:
        print("No cross-symbol daily inconsistencies found.")
    print()

    # Per-symbol yearly session summary
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
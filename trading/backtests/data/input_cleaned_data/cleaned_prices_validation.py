import math
import pandas as pd
from pathlib import Path

YEAR = 2023

# This script is assumed to live in: data/cleaned_data/
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / f"cleaned_prices_{YEAR}.csv"
INCONSISTENCY_FILE = BASE_DIR / f"cleaned_prices_{YEAR}_inconsistencies.csv"

DOMINANT_PATTERN_RATIO = 2 / 3
EXPECTED_TOTAL_SYMBOLS = 15  # fixed study universe


def main():
    df = pd.read_csv(INPUT_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    print("Rows:", len(df))
    print("Symbols:", df["symbol"].nunique())
    print("Min time:", df["time"].min())
    print("Max time:", df["time"].max())
    print()

    dupes = df.duplicated(subset=["symbol", "time"]).sum()
    print("Duplicate symbol+time rows:", dupes)

    bad_order_symbols = []
    for symbol, g in df.groupby("symbol"):
        if not g["time"].is_monotonic_increasing:
            bad_order_symbols.append(symbol)

    print("Symbols with non-monotonic time order:", len(bad_order_symbols))
    if bad_order_symbols:
        print("Bad order symbols:", bad_order_symbols)
    print()

    core_cols = [
        "symbol", "time", "open", "high", "low", "close",
        "tick_volume", "spread", "real_volume"
    ]
    missing_rows = df[core_cols].isna().any(axis=1).sum()
    print("Rows with missing core fields:", missing_rows)

    bad_ohlc = df[
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df["open"] <= 0) |
        (df["high"] <= 0) |
        (df["low"] <= 0) |
        (df["close"] <= 0)
    ]
    print("Rows with invalid OHLC structure:", len(bad_ohlc))

    bad_nonnegative = df[
        (df["tick_volume"] < 0) |
        (df["spread"] < 0) |
        (df["real_volume"] < 0)
    ]
    print("Rows with negative volume/spread fields:", len(bad_nonnegative))

    bad_grid = df[
        (df["time"].dt.second != 0) |
        (df["time"].dt.microsecond != 0) |
        (df["time"].dt.minute % 5 != 0)
    ]
    print("Rows off the 5-minute grid:", len(bad_grid))
    print()

    df["date"] = df["time"].dt.date

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

    dominant_count_threshold = math.ceil(EXPECTED_TOTAL_SYMBOLS * DOMINANT_PATTERN_RATIO)
    inconsistent_dates = []

    # Validate cleaned data according to the dominant-pattern rule:
    # keep dates where at least threshold remaining symbols share one full timestamp sequence.
    for date, group in df.groupby("date"):
        group = group.sort_values(["symbol", "time"]).copy()

        symbol_times = {}
        for symbol, sg in group.groupby("symbol"):
            timestamps = tuple(
                sg.sort_values("time")["time"].dt.strftime("%H:%M:%S").tolist()
            )
            symbol_times[symbol] = timestamps

        if not symbol_times:
            continue

        pattern_to_symbols = {}
        for symbol, times in symbol_times.items():
            pattern_to_symbols.setdefault(times, []).append(symbol)

        dominant_pattern, dominant_symbols = max(
            pattern_to_symbols.items(),
            key=lambda item: len(item[1])
        )
        dominant_count = len(dominant_symbols)

        deviating_symbols = sorted(
            symbol for symbol, times in symbol_times.items()
            if times != dominant_pattern
        )

        date_is_inconsistent = dominant_count < dominant_count_threshold or len(deviating_symbols) > 0

        if date_is_inconsistent:
            inconsistent_daily = daily[daily["date"] == date].copy()
            inconsistent_daily["timestamp_mismatch"] = False

            if deviating_symbols:
                inconsistent_daily.loc[
                    inconsistent_daily["symbol"].isin(deviating_symbols),
                    "timestamp_mismatch"
                ] = True

            inconsistent_daily["retained_symbols_count"] = len(symbol_times)
            inconsistent_daily["dominant_pattern_count"] = dominant_count
            inconsistent_daily["required_threshold"] = dominant_count_threshold
            inconsistent_daily["deviating_symbols"] = ", ".join(deviating_symbols)

            inconsistent_dates.append(inconsistent_daily)

    if inconsistent_dates:
        result = pd.concat(inconsistent_dates, ignore_index=True)
        result.to_csv(INCONSISTENCY_FILE, index=False)
        print(f"Saved inconsistent dates to {INCONSISTENCY_FILE}")
        print("Number of inconsistent dates:", result["date"].nunique())
        print("Number of symbol-date rows written:", len(result))
    else:
        print("No post-cleaning inconsistencies found.")
    print()

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
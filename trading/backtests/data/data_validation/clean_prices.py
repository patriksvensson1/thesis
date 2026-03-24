import math
import pandas as pd
from pathlib import Path

YEAR = 2024

# This script is assumed to live in: data/data_validation/
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent

RAW_DATA_DIR = DATA_DIR / "raw_data"
VALIDATION_DIR = DATA_DIR / "data_validation"
CLEANED_DIR = DATA_DIR / "cleaned_data"

RAW_PRICE_FILE = RAW_DATA_DIR / f"raw_price_data_{YEAR}.csv"
INCONSISTENCY_FILE = VALIDATION_DIR / f"raw_price_data_{YEAR}_inconsistencies.csv"
CLEANED_FILE = CLEANED_DIR / f"cleaned_prices_{YEAR}.csv"

# Require at least two thirds of symbols to share one pattern
DOMINANT_PATTERN_RATIO = 2 / 3


def main():
    # Make sure the cleaned-data folder exists before saving output
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw prices and the cross-sectional inconsistency report
    prices = pd.read_csv(RAW_PRICE_FILE)
    bad = pd.read_csv(INCONSISTENCY_FILE)

    # Parse timestamps / dates
    prices["time"] = pd.to_datetime(prices["time"], utc=True)
    bad["date"] = pd.to_datetime(bad["date"]).dt.date

    # Remove duplicate rows from the inconsistency file so symbols do not get printed repeatedly
    bad = bad.drop_duplicates(
        subset=["date", "symbol", "bars", "first_bar_str", "last_bar_str"]
    ).copy()

    # Create trading date in raw prices
    prices["date"] = prices["time"].dt.date

    # Total number of symbols expected in the universe
    total_symbols = prices["symbol"].nunique()
    dominant_count_threshold = math.ceil(total_symbols * DOMINANT_PATTERN_RATIO)

    # We will either:
    # 1. remove only specific symbol-date pairs, or
    # 2. remove an entire date if no strong dominant pattern exists
    remove_pairs = set()
    remove_dates = set()

    print("Cleaning inconsistent dates:")
    print(
        f"Dominant-pattern rule: keep a date only if at least "
        f"{dominant_count_threshold}/{total_symbols} symbols share one pattern."
    )

    for date, group in bad.groupby("date"):
        # Count how many symbols share each observed pattern on this date
        pattern_counts = (
            group.groupby(["bars", "first_bar_str", "last_bar_str"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "bars"], ascending=[False, False])
            .reset_index(drop=True)
        )

        dominant = pattern_counts.iloc[0]
        dominant_pattern = (
            dominant["bars"],
            dominant["first_bar_str"],
            dominant["last_bar_str"],
        )
        dominant_count = int(dominant["count"])

        print(
            f"\nDate {date} | dominant pattern: "
            f"bars={dominant_pattern[0]}, "
            f"first={dominant_pattern[1]}, "
            f"last={dominant_pattern[2]}, "
            f"count={dominant_count}/{total_symbols}"
        )

        # If the dominant pattern is not strong enough, drop the whole date
        if dominant_count < dominant_count_threshold:
            remove_dates.add(date)
            print(f"  DROP ENTIRE DATE {date} | no strong dominant pattern")
            continue

        # Otherwise, keep the dominant pattern and remove only deviating symbols
        date_removals = set()

        for _, row in group.iterrows():
            row_pattern = (row["bars"], row["first_bar_str"], row["last_bar_str"])
            if row_pattern != dominant_pattern:
                remove_pairs.add((row["symbol"], row["date"]))
                date_removals.add(
                    (
                        row["symbol"],
                        row["date"],
                        row["bars"],
                        row["first_bar_str"],
                        row["last_bar_str"],
                    )
                )

        if not date_removals:
            print("  No symbol removals needed after dominant-pattern check")
        else:
            for symbol, row_date, bars, first_bar, last_bar in sorted(date_removals):
                print(
                    f"  REMOVE {row_date} | {symbol} | "
                    f"bars={bars} first={first_bar} last={last_bar}"
                )

    # Remove entire bad dates first
    cleaned = prices[~prices["date"].isin(remove_dates)].copy()

    # Then remove symbol-date pairs that deviated from valid dominant patterns
    cleaned = cleaned[
        ~cleaned.apply(lambda row: (row["symbol"], row["date"]) in remove_pairs, axis=1)
    ].copy()

    # Drop helper date column
    cleaned = cleaned.drop(columns=["date"])

    # Sort for reproducibility
    cleaned = cleaned.sort_values(["symbol", "time"]).reset_index(drop=True)

    # Save cleaned prices
    cleaned.to_csv(CLEANED_FILE, index=False)

    # Print summary
    removed_rows = len(prices) - len(cleaned)

    print()
    print(f"Raw rows: {len(prices)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Removed rows: {removed_rows}")
    print(f"Removed full dates: {len(remove_dates)}")
    if remove_dates:
        print("Dropped dates:", sorted(remove_dates))
    print(f"Removed symbol-date observations: {len(remove_pairs)}")
    print(f"Loaded raw file from: {RAW_PRICE_FILE}")
    print(f"Loaded inconsistency file from: {INCONSISTENCY_FILE}")
    print(f"Saved cleaned prices to: {CLEANED_FILE}")


if __name__ == "__main__":
    main()
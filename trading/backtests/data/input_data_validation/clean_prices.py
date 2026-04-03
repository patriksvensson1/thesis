import math
import pandas as pd
from pathlib import Path

YEAR = 2024

# This script is assumed to live in: data/data_validation/
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent

RAW_DATA_DIR = DATA_DIR / "input_raw_data"
VALIDATION_DIR = DATA_DIR / "input_data_validation"
CLEANED_DIR = DATA_DIR / "input_cleaned_data"

RAW_PRICE_FILE = RAW_DATA_DIR / f"raw_price_data_{YEAR}.csv"
INCONSISTENCY_FILE = VALIDATION_DIR / f"raw_price_data_{YEAR}_inconsistencies.csv"
CLEANED_FILE = CLEANED_DIR / f"cleaned_prices_{YEAR}.csv"

# Require at least two thirds of symbols to share one pattern
DOMINANT_PATTERN_RATIO = 2 / 3


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw prices and the inconsistency report
    prices = pd.read_csv(RAW_PRICE_FILE)
    bad = pd.read_csv(INCONSISTENCY_FILE)

    # Parse timestamps / dates
    prices["time"] = pd.to_datetime(prices["time"], utc=True)
    prices["date"] = prices["time"].dt.date
    bad["date"] = pd.to_datetime(bad["date"]).dt.date

    # Remove exact duplicate rows in inconsistency file
    bad = bad.drop_duplicates().copy()

    # Total number of symbols expected in the universe
    all_symbols = sorted(prices["symbol"].unique())
    total_symbols = len(all_symbols)
    dominant_count_threshold = math.ceil(total_symbols * DOMINANT_PATTERN_RATIO)

    remove_pairs = set()
    remove_dates = set()

    print("Cleaning inconsistent dates:")
    print(
        f"Dominant-pattern rule: keep a date only if at least "
        f"{dominant_count_threshold}/{total_symbols} symbols share one pattern."
    )

    for date, group in bad.groupby("date"):
        date_prices = prices[prices["date"] == date].copy()

        present_symbols = sorted(date_prices["symbol"].unique())
        missing_symbols = sorted(set(all_symbols) - set(present_symbols))

        # Drop the full date only if more than 5 symbols are completely missing.
        # Otherwise, let the dominant-pattern rule decide based on the symbols present.
        if len(missing_symbols) > 5:
            remove_dates.add(date)
            print(
                f"\nDate {date} | DROP ENTIRE DATE | "
                f"missing_symbols_count={len(missing_symbols)} | "
                f"missing_symbols={', '.join(missing_symbols)}"
            )
            continue

        # Build full ordered timestamp sequence for each symbol on this date
        symbol_patterns = {}
        for symbol, sg in date_prices.groupby("symbol"):
            pattern = tuple(
                sg.sort_values("time")["time"].dt.strftime("%H:%M:%S").tolist()
            )
            symbol_patterns[symbol] = pattern

        # Count how many symbols share each full timestamp pattern
        pattern_to_symbols = {}
        for symbol, pattern in symbol_patterns.items():
            pattern_to_symbols.setdefault(pattern, []).append(symbol)

        # Sort by count descending, then by pattern length descending
        ranked_patterns = sorted(
            pattern_to_symbols.items(),
            key=lambda item: (len(item[1]), len(item[0])),
            reverse=True,
        )

        dominant_pattern, dominant_symbols = ranked_patterns[0]
        dominant_count = len(dominant_symbols)

        print(
            f"\nDate {date} | dominant pattern: "
            f"bars={len(dominant_pattern)}, "
            f"first={dominant_pattern[0] if dominant_pattern else ''}, "
            f"last={dominant_pattern[-1] if dominant_pattern else ''}, "
            f"count={dominant_count}/{total_symbols}"
        )

        # If the dominant full-sequence pattern is not strong enough, drop whole date
        if dominant_count < dominant_count_threshold:
            remove_dates.add(date)
            print(f"  DROP ENTIRE DATE {date} | no strong dominant pattern")
            continue

        # Otherwise, remove only symbols that deviate from dominant full sequence
        date_removals = set()
        for symbol, pattern in symbol_patterns.items():
            if pattern != dominant_pattern:
                remove_pairs.add((symbol, date))
                date_removals.add(
                    (
                        symbol,
                        date,
                        len(pattern),
                        pattern[0] if pattern else "",
                        pattern[-1] if pattern else "",
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
        dropped_dates_str = [d.isoformat() for d in sorted(remove_dates)]
        print("Dropped dates:", dropped_dates_str)
    print(f"Removed symbol-date observations: {len(remove_pairs)}")
    print(f"Loaded raw file from: {RAW_PRICE_FILE}")
    print(f"Loaded inconsistency file from: {INCONSISTENCY_FILE}")
    print(f"Saved cleaned prices to: {CLEANED_FILE}")


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path

YEAR = 2022

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
RAW_DIR = DATA_DIR / "raw_data"

NEWS_FILE = RAW_DIR / f"raw_gdelt_news_{YEAR}.csv"
DUPLICATES_FILE = BASE_DIR / f"raw_gdelt_news_{YEAR}_duplicates.csv"
OUT_OF_RANGE_FILE = BASE_DIR / f"raw_gdelt_news_{YEAR}_out_of_range.csv"


def main():
    # Load article file from ../raw_data
    news = pd.read_csv(NEWS_FILE)

    print("=== NEWS FILE CHECKS ===")
    print("Rows:", len(news))
    print("Symbols:", news["symbol"].nunique())
    print()

    # Check missing values in key fields
    print("Missing symbol:", news["symbol"].isna().sum())
    print("Missing url:", news["url"].isna().sum())
    print("Missing seendate:", news["seendate"].isna().sum())
    print()

    # True duplicate checks: only treat exact same symbol + url as duplicates
    dupes_symbol_url = news.duplicated(subset=["symbol", "url"]).sum()
    print("Duplicate symbol+url rows:", dupes_symbol_url)

    dupes_symbol_url_time = news.duplicated(subset=["symbol", "url", "seendate"]).sum()
    print("Duplicate symbol+url+seendate rows:", dupes_symbol_url_time)
    print()

    # Parse GDELT seendate into UTC datetimes
    news["parsed_seendate"] = pd.to_datetime(
        news["seendate"],
        format="%Y%m%dT%H%M%SZ",
        utc=True,
        errors="coerce"
    )

    bad_dates = news["parsed_seendate"].isna().sum()
    print("Rows with unparseable seendate:", bad_dates)

    # Check that parsed timestamps fall inside the requested year
    year_start = pd.Timestamp(f"{YEAR}-01-01 00:00:00", tz="UTC")
    year_end = pd.Timestamp(f"{YEAR + 1}-01-01 00:00:00", tz="UTC")

    out_of_range = news[
        (news["parsed_seendate"].notna()) &
        (
            (news["parsed_seendate"] < year_start) |
            (news["parsed_seendate"] >= year_end)
        )
    ].copy()
    print("Rows with seendate outside requested year:", len(out_of_range))
    print()

    # Simple article count summary by symbol
    article_counts = (
        news.groupby("symbol")
        .size()
        .reset_index(name="article_rows")
        .sort_values("article_rows", ascending=False)
    )

    print("=== ARTICLE COUNTS BY SYMBOL ===")
    print(article_counts.to_string(index=False))
    print()

    # Save duplicate rows into the same folder as the script
    duplicate_rows = news[news.duplicated(subset=["symbol", "url"], keep=False)].copy()

    if not duplicate_rows.empty:
        duplicate_rows = duplicate_rows.sort_values(
            by=["symbol", "url", "parsed_seendate", "title"],
            na_position="last"
        )

        duplicate_rows.to_csv(DUPLICATES_FILE, index=False)
        print(f"Saved duplicates to {DUPLICATES_FILE}")

    # Save out-of-range rows into the same folder as the script
    if not out_of_range.empty:
        out_of_range = out_of_range.sort_values(
            by=["symbol", "parsed_seendate", "url"],
            na_position="last"
        )
        out_of_range.to_csv(OUT_OF_RANGE_FILE, index=False)
        print(f"Saved out-of-range rows to {OUT_OF_RANGE_FILE}")


if __name__ == "__main__":
    main()
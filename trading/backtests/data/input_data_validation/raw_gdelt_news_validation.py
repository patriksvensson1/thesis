import pandas as pd
from pathlib import Path

YEAR = 2023

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent
RAW_DIR = DATA_DIR / "input_raw_data"

INPUT_FILES = [
    RAW_DIR / f"raw_gdelt_news_{YEAR}_1.csv",
    RAW_DIR / f"raw_gdelt_news_{YEAR}_2.csv",
]


def process_file(news_file: Path):
    suffix = news_file.stem.split("_")[-1]

    duplicates_file = BASE_DIR / f"raw_news_{YEAR}_{suffix}_duplicates.csv"
    out_of_range_file = BASE_DIR / f"raw_news_{YEAR}_{suffix}_out_of_range.csv"
    timestamp_mismatch_file = BASE_DIR / f"raw_news_{YEAR}_{suffix}_timestamp_mismatches.csv"

    news = pd.read_csv(news_file)

    print(f"=== NEWS FILE CHECKS: {news_file.name} ===")
    print("Rows:", len(news))
    print("Symbols:", news["symbol"].nunique())
    print()

    print("Missing symbol:", news["symbol"].isna().sum())
    print("Missing url:", news["url"].isna().sum())
    print("Missing seendate:", news["seendate"].isna().sum())
    print("Missing seen_at_utc:", news["seen_at_utc"].isna().sum() if "seen_at_utc" in news.columns else "COLUMN MISSING")
    print()

    dupes_symbol_url = news.duplicated(subset=["symbol", "url"]).sum()
    print("Duplicate symbol+url rows:", dupes_symbol_url)

    dupes_symbol_url_time = news.duplicated(subset=["symbol", "url", "seendate"]).sum()
    print("Duplicate symbol+url+seendate rows:", dupes_symbol_url_time)
    print()

    news["parsed_seendate"] = pd.to_datetime(
        news["seendate"],
        format="%Y%m%dT%H%M%SZ",
        utc=True,
        errors="coerce"
    )

    bad_seendate = news["parsed_seendate"].isna().sum()
    print("Rows with unparseable seendate:", bad_seendate)

    if "seen_at_utc" in news.columns:
        news["parsed_seen_at_utc"] = pd.to_datetime(
            news["seen_at_utc"],
            utc=True,
            errors="coerce"
        )
        bad_seen_at_utc = news["parsed_seen_at_utc"].isna().sum()
        print("Rows with unparseable seen_at_utc:", bad_seen_at_utc)
    else:
        news["parsed_seen_at_utc"] = pd.NaT
        print("Rows with unparseable seen_at_utc: COLUMN MISSING")

    print()

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

    if "seen_at_utc" in news.columns:
        out_of_range_seen_at = news[
            (news["parsed_seen_at_utc"].notna()) &
            (
                (news["parsed_seen_at_utc"] < year_start) |
                (news["parsed_seen_at_utc"] >= year_end)
            )
        ].copy()
        print("Rows with seen_at_utc outside requested year:", len(out_of_range_seen_at))
    else:
        print("Rows with seen_at_utc outside requested year: COLUMN MISSING")

    print()

    if "seen_at_utc" in news.columns:
        timestamp_mismatches = news[
            (news["parsed_seendate"].notna()) &
            (news["parsed_seen_at_utc"].notna()) &
            (news["parsed_seendate"] != news["parsed_seen_at_utc"])
        ].copy()
        print("Rows where seendate != seen_at_utc:", len(timestamp_mismatches))
    else:
        timestamp_mismatches = pd.DataFrame()
        print("Rows where seendate != seen_at_utc: COLUMN MISSING")

    print()

    article_counts = (
        news.groupby("symbol")
        .size()
        .reset_index(name="article_rows")
        .sort_values("article_rows", ascending=False)
    )

    print("=== ARTICLE COUNTS BY SYMBOL ===")
    print(article_counts.to_string(index=False))
    print()

    duplicate_rows = news[news.duplicated(subset=["symbol", "url"], keep=False)].copy()

    if not duplicate_rows.empty:
        duplicate_rows = duplicate_rows.sort_values(
            by=["symbol", "url", "parsed_seendate", "title"],
            na_position="last"
        )
        duplicate_rows.to_csv(duplicates_file, index=False)
        print(f"Saved duplicates to {duplicates_file}")

    if not out_of_range.empty:
        out_of_range = out_of_range.sort_values(
            by=["symbol", "parsed_seendate", "url"],
            na_position="last"
        )
        out_of_range.to_csv(out_of_range_file, index=False)
        print(f"Saved out-of-range rows to {out_of_range_file}")

    if not timestamp_mismatches.empty:
        timestamp_mismatches = timestamp_mismatches.sort_values(
            by=["symbol", "parsed_seendate", "url"],
            na_position="last"
        )
        timestamp_mismatches.to_csv(timestamp_mismatch_file, index=False)
        print(f"Saved timestamp mismatches to {timestamp_mismatch_file}")

    print()


def main():
    for news_file in INPUT_FILES:
        if news_file.exists():
            process_file(news_file)
        else:
            print(f"File not found: {news_file}")


if __name__ == "__main__":
    main()
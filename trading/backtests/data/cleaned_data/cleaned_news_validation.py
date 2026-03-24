from pathlib import Path
import pandas as pd

YEAR = 2022

BASE_DIR = Path(__file__).resolve().parent
CLEANED_FILE = BASE_DIR / f"cleaned_news_{YEAR}.csv"
DUPLICATES_OUT = BASE_DIR / f"cleaned_news_{YEAR}_remaining_duplicates.csv"
OUT_OF_RANGE_OUT = BASE_DIR / f"cleaned_news_{YEAR}_remaining_out_of_range.csv"


def main():
    news = pd.read_csv(CLEANED_FILE)

    print("=== CLEANED NEWS FILE CHECKS ===")
    print("Rows:", len(news))
    print("Symbols:", news["symbol"].nunique())
    print()

    print("Missing symbol:", news["symbol"].isna().sum())
    print("Missing url:", news["url"].isna().sum())
    print("Missing seendate:", news["seendate"].isna().sum())
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

    bad_dates = news["parsed_seendate"].isna().sum()
    print("Rows with unparseable seendate:", bad_dates)

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

    article_counts = (
        news.groupby("symbol")
        .size()
        .reset_index(name="article_rows")
        .sort_values("article_rows", ascending=False)
    )

    print("=== ARTICLE COUNTS BY SYMBOL ===")
    print(article_counts.to_string(index=False))
    print()

    if dupes_symbol_url == 0 and dupes_symbol_url_time == 0 and len(out_of_range) == 0 and bad_dates == 0:
        print("Validation passed: no duplicates, no out-of-range rows, no bad timestamps.")
    else:
        print("Validation failed: inspect the counts above.")

        if dupes_symbol_url > 0:
            duplicate_rows = news[news.duplicated(subset=["symbol", "url"], keep=False)].copy()
            duplicate_rows = duplicate_rows.sort_values(
                by=["symbol", "url", "parsed_seendate", "title"],
                na_position="last"
            )
            duplicate_rows.to_csv(DUPLICATES_OUT, index=False)
            print(f"Saved remaining duplicates to {DUPLICATES_OUT}")

        if len(out_of_range) > 0:
            out_of_range = out_of_range.sort_values(
                by=["symbol", "parsed_seendate", "url"],
                na_position="last"
            )
            out_of_range.to_csv(OUT_OF_RANGE_OUT, index=False)
            print(f"Saved remaining out-of-range rows to {OUT_OF_RANGE_OUT}")


if __name__ == "__main__":
    main()
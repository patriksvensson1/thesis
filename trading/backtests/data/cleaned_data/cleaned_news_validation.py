from pathlib import Path
import pandas as pd

YEAR = 2022

BASE_DIR = Path(__file__).resolve().parent
CLEANED_FILE = BASE_DIR / f"cleaned_news_{YEAR}.csv"
DUPLICATES_OUT = BASE_DIR / f"cleaned_news_{YEAR}_remaining_duplicates.csv"
OUT_OF_RANGE_OUT = BASE_DIR / f"cleaned_news_{YEAR}_remaining_out_of_range.csv"
TIMESTAMP_MISMATCH_OUT = BASE_DIR / f"cleaned_news_{YEAR}_remaining_timestamp_mismatches.csv"


def main():
    news = pd.read_csv(CLEANED_FILE)

    print("=== CLEANED NEWS FILE CHECKS ===")
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
        bad_seen_at_utc = len(news)
        print("Rows with unparseable seen_at_utc: COLUMN MISSING")

    print()

    allowed_start = pd.Timestamp(f"{YEAR-1}-12-31 00:00:00", tz="UTC")
    allowed_end = pd.Timestamp(f"{YEAR + 1}-01-01 00:00:00", tz="UTC")

    out_of_range = news[
        (news["parsed_seendate"].notna()) &
        (
            (news["parsed_seendate"] < allowed_start) |
            (news["parsed_seendate"] >= allowed_end)
        )
    ].copy()

    print("Rows with seendate outside allowed range:", len(out_of_range))

    if "seen_at_utc" in news.columns:
        out_of_range_seen_at = news[
            (news["parsed_seen_at_utc"].notna()) &
            (
                (news["parsed_seen_at_utc"] < allowed_start) |
                (news["parsed_seen_at_utc"] >= allowed_end)
            )
        ].copy()
        print("Rows with seen_at_utc outside allowed range:", len(out_of_range_seen_at))
    else:
        out_of_range_seen_at = pd.DataFrame()
        print("Rows with seen_at_utc outside allowed range: COLUMN MISSING")

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

    validation_ok = (
        dupes_symbol_url == 0 and
        dupes_symbol_url_time == 0 and
        len(out_of_range) == 0 and
        len(out_of_range_seen_at) == 0 and
        len(timestamp_mismatches) == 0 and
        bad_seendate == 0 and
        bad_seen_at_utc == 0
    )

    if validation_ok:
        print("Validation passed: no duplicates, no out-of-range rows, no bad timestamps, no seendate/seen_at_utc mismatches.")
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

        if len(timestamp_mismatches) > 0:
            timestamp_mismatches = timestamp_mismatches.sort_values(
                by=["symbol", "parsed_seendate", "url"],
                na_position="last"
            )
            timestamp_mismatches.to_csv(TIMESTAMP_MISMATCH_OUT, index=False)
            print(f"Saved remaining timestamp mismatches to {TIMESTAMP_MISMATCH_OUT}")


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path

YEAR = 2023

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent

RAW_DIR = DATA_DIR / "raw_data"
CLEANED_DIR = DATA_DIR / "cleaned_data"

RAW_FILE = RAW_DIR / f"raw_gdelt_news_{YEAR}.csv"
DUPLICATES_FILE = BASE_DIR / f"raw_news_{YEAR}_duplicates.csv"
OUT_OF_RANGE_FILE = BASE_DIR / f"raw_news_{YEAR}_out_of_range.csv"
TIMESTAMP_MISMATCH_FILE = BASE_DIR / f"raw_news_{YEAR}_timestamp_mismatches.csv"
OUTPUT_FILE = CLEANED_DIR / f"cleaned_news_{YEAR}.csv"


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    news = pd.read_csv(RAW_FILE)
    print(f"Loaded raw file: {RAW_FILE} | rows={len(news)}")

    required_cols = {"symbol", "title", "url", "seendate"}
    missing = required_cols - set(news.columns)
    if missing:
        raise ValueError(f"Raw file is missing required columns: {sorted(missing)}")

    if "seen_at_utc" not in news.columns:
        news["seen_at_utc"] = pd.NA
        print("Column seen_at_utc missing in raw file; created empty column.")

    news["symbol"] = news["symbol"].astype(str)
    news["url"] = news["url"].astype(str)
    news["seendate"] = news["seendate"].astype(str)

    news["parsed_seendate"] = pd.to_datetime(
        news["seendate"],
        format="%Y%m%dT%H%M%SZ",
        utc=True,
        errors="coerce"
    )

    news["parsed_seen_at_utc"] = pd.to_datetime(
        news["seen_at_utc"],
        utc=True,
        errors="coerce"
    )

    before = len(news)
    news = news[news["parsed_seendate"].notna()].copy()
    removed_bad_seendate = before - len(news)
    print(f"Removed rows with invalid seendate: {removed_bad_seendate}")

    rebuild_mask = news["parsed_seen_at_utc"].isna()
    rebuilt_count = int(rebuild_mask.sum())
    if rebuilt_count > 0:
        news.loc[rebuild_mask, "parsed_seen_at_utc"] = news.loc[rebuild_mask, "parsed_seendate"]
    print(f"Rebuilt seen_at_utc from seendate for rows: {rebuilt_count}")

    if OUT_OF_RANGE_FILE.exists():
        out_of_range = pd.read_csv(OUT_OF_RANGE_FILE)

        out_keys = set(
            zip(
                out_of_range["symbol"].astype(str),
                out_of_range["url"].astype(str),
                out_of_range["seendate"].astype(str),
            )
        )

        before = len(news)
        news = news[
            ~news.apply(
                lambda row: (
                    str(row["symbol"]),
                    str(row["url"]),
                    str(row["seendate"])
                ) in out_keys,
                axis=1
            )
        ].copy()
        removed = before - len(news)
        print(f"Removed out-of-range rows: {removed}")
    else:
        print(f"No out-of-range file found: {OUT_OF_RANGE_FILE}")

    if TIMESTAMP_MISMATCH_FILE.exists():
        mismatch_rows = pd.read_csv(TIMESTAMP_MISMATCH_FILE)

        mismatch_keys = set(
            zip(
                mismatch_rows["symbol"].astype(str),
                mismatch_rows["url"].astype(str),
                mismatch_rows["seendate"].astype(str),
            )
        )

        before = len(news)
        news = news[
            ~news.apply(
                lambda row: (
                    str(row["symbol"]),
                    str(row["url"]),
                    str(row["seendate"])
                ) in mismatch_keys,
                axis=1
            )
        ].copy()
        removed = before - len(news)
        print(f"Removed timestamp mismatch rows: {removed}")
    else:
        print(f"No timestamp mismatch file found: {TIMESTAMP_MISMATCH_FILE}")

    year_start = pd.Timestamp(f"{YEAR}-01-01 00:00:00", tz="UTC")
    year_end = pd.Timestamp(f"{YEAR+1}-01-01 00:00:00", tz="UTC")

    before = len(news)
    news = news[
        (news["parsed_seendate"] >= year_start) &
        (news["parsed_seendate"] < year_end)
    ].copy()
    removed = before - len(news)
    print(f"Removed rows outside requested year after filtering: {removed}")

    before = len(news)
    news = news.sort_values(
        by=["symbol", "url", "parsed_seendate", "title"],
        na_position="last"
    ).drop_duplicates(subset=["symbol", "url"], keep="first").copy()
    removed = before - len(news)
    print(f"Removed duplicate symbol+url rows: {removed}")

    if DUPLICATES_FILE.exists():
        duplicate_rows = pd.read_csv(DUPLICATES_FILE)
        print(f"Duplicate reference file found: {DUPLICATES_FILE} | rows={len(duplicate_rows)}")
    else:
        print(f"No duplicate reference file found: {DUPLICATES_FILE}")

    news["seen_at_utc"] = news["parsed_seen_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    news["seen_at_utc"] = news["seen_at_utc"].str.replace(r"(\+0000)$", "+00:00", regex=True)

    keep_columns = [
        "symbol",
        "title",
        "url",
        "domain",
        "seendate",
        "seen_at_utc",
        "language",
        "sourcecountry",
    ]

    for col in keep_columns:
        if col not in news.columns:
            news[col] = pd.NA

    news = news[keep_columns].copy()
    news = news.sort_values(["symbol", "seen_at_utc", "url"], na_position="last").reset_index(drop=True)

    news.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned file: {OUTPUT_FILE} | rows={len(news)}")


if __name__ == "__main__":
    main()
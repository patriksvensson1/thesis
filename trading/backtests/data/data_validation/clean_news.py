import pandas as pd
from pathlib import Path

YEAR = 2022

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent

RAW_DIR = DATA_DIR / "raw_data"
CLEANED_DIR = DATA_DIR / "cleaned_data"

RAW_FILE = RAW_DIR / f"raw_gdelt_news_{YEAR}.csv"
DUPLICATES_FILE = BASE_DIR / f"raw_gdelt_news_{YEAR}_duplicates.csv"
OUT_OF_RANGE_FILE = BASE_DIR / f"raw_gdelt_news_{YEAR}_out_of_range.csv"
OUTPUT_FILE = CLEANED_DIR / f"cleaned_news_{YEAR}.csv"


def main():
    # Make sure cleaned_data folder exists
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw news data from ../raw_data
    news = pd.read_csv(RAW_FILE)
    print(f"Loaded raw file: {RAW_FILE} | rows={len(news)}")

    # Parse seendate consistently so we can sort and match rows safely
    news["parsed_seendate"] = pd.to_datetime(
        news["seendate"],
        format="%Y%m%dT%H%M%SZ",
        utc=True,
        errors="coerce"
    )

    # Remove out-of-range rows if that file exists
    if OUT_OF_RANGE_FILE.exists():
        out_of_range = pd.read_csv(OUT_OF_RANGE_FILE)

        # Match rows using symbol + url + seendate
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

    # Remove duplicate rows by keeping only the earliest occurrence per symbol + url
    before = len(news)
    news = news.sort_values(
        by=["symbol", "url", "parsed_seendate", "title"],
        na_position="last"
    ).drop_duplicates(subset=["symbol", "url"], keep="first").copy()
    removed = before - len(news)
    print(f"Removed duplicate symbol+url rows: {removed}")

    # Optional reference check
    if DUPLICATES_FILE.exists():
        duplicate_rows = pd.read_csv(DUPLICATES_FILE)
        print(f"Duplicate reference file found: {DUPLICATES_FILE} | rows={len(duplicate_rows)}")
    else:
        print(f"No duplicate reference file found: {DUPLICATES_FILE}")

    # Drop helper column before saving
    news = news.drop(columns=["parsed_seendate"], errors="ignore")

    # Save cleaned file to ../cleaned_data
    news.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned file: {OUTPUT_FILE} | rows={len(news)}")


if __name__ == "__main__":
    main()
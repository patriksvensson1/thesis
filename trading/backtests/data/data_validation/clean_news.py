import pandas as pd
from pathlib import Path

YEAR = 2024

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent

RAW_DIR = DATA_DIR / "raw_data"
CLEANED_DIR = DATA_DIR / "cleaned_data"

INPUT_FILES = [
    RAW_DIR / f"raw_gdelt_news_{YEAR}_1.csv",
    RAW_DIR / f"raw_gdelt_news_{YEAR}_2.csv",
]


def remove_rows_by_reference(news: pd.DataFrame, reference_file: Path, label: str) -> pd.DataFrame:
    if reference_file.exists():
        ref = pd.read_csv(reference_file)

        required_cols = {"symbol", "url", "seendate"}
        missing = required_cols - set(ref.columns)
        if missing:
            print(f"Reference file {reference_file} missing columns {sorted(missing)}; skipping {label} removal.")
            return news

        ref_keys = set(
            zip(
                ref["symbol"].astype(str),
                ref["url"].astype(str),
                ref["seendate"].astype(str),
            )
        )

        before = len(news)
        news = news[
            ~news.apply(
                lambda row: (
                    str(row["symbol"]),
                    str(row["url"]),
                    str(row["seendate"])
                ) in ref_keys,
                axis=1
            )
        ].copy()
        removed = before - len(news)
        print(f"Removed {label} rows: {removed}")
    else:
        print(f"No {label} reference file found: {reference_file}")

    return news


def process_file(raw_file: Path):
    suffix = raw_file.stem.split("_")[-1]

    duplicates_file = BASE_DIR / f"raw_news_{YEAR}_{suffix}_duplicates.csv"
    out_of_range_file = BASE_DIR / f"raw_news_{YEAR}_{suffix}_out_of_range.csv"
    timestamp_mismatch_file = BASE_DIR / f"raw_news_{YEAR}_{suffix}_timestamp_mismatches.csv"
    output_file = CLEANED_DIR / f"cleaned_news_{YEAR}_{suffix}.csv"

    news = pd.read_csv(raw_file)
    print(f"Loaded raw file: {raw_file} | rows={len(news)}")

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

    before = len(news)
    news = news[news["parsed_seendate"].notna()].copy()
    removed_bad_seendate = before - len(news)
    print(f"Removed rows with invalid seendate: {removed_bad_seendate}")

    news["parsed_seen_at_utc"] = pd.to_datetime(
        news["seen_at_utc"],
        utc=True,
        errors="coerce"
    )

    rebuild_mask = news["parsed_seen_at_utc"].isna()
    rebuilt_count = int(rebuild_mask.sum())
    if rebuilt_count > 0:
        news.loc[rebuild_mask, "parsed_seen_at_utc"] = news.loc[rebuild_mask, "parsed_seendate"]
    print(f"Rebuilt seen_at_utc from seendate for rows: {rebuilt_count}")

    news = remove_rows_by_reference(news, out_of_range_file, "out-of-range")
    news = remove_rows_by_reference(news, timestamp_mismatch_file, "timestamp mismatch")

    year_start = pd.Timestamp(f"{YEAR}-01-01 00:00:00", tz="UTC")
    year_end = pd.Timestamp(f"{YEAR + 1}-01-01 00:00:00", tz="UTC")

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

    if duplicates_file.exists():
        duplicate_rows = pd.read_csv(duplicates_file)
        print(f"Duplicate reference file found: {duplicates_file} | rows={len(duplicate_rows)}")
    else:
        print(f"No duplicate reference file found: {duplicates_file}")

    news["seen_at_utc"] = news["parsed_seen_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    news["seen_at_utc"] = news["seen_at_utc"].str.replace(r"(\+0000)$", "Z", regex=True)

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

    news.to_csv(output_file, index=False)
    print(f"Saved cleaned file: {output_file} | rows={len(news)}")
    print()


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    for raw_file in INPUT_FILES:
        if raw_file.exists():
            process_file(raw_file)
        else:
            print(f"File not found: {raw_file}")


if __name__ == "__main__":
    main()
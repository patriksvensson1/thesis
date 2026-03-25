import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

YEAR = 2023
START_DATE = datetime(YEAR - 1, 12, 31)
END_DATE = datetime(YEAR + 1, 1, 1)

RATE_LIMIT = 6
RETRY_SLEEP = 6
MAX_JSON_ERRORS_PER_DAY = 10

OUTPUT_FILE = BASE_DIR / f"raw_gdelt_news_{YEAR}.csv"
PROGRESS_FILE = BASE_DIR / f"gdelt_progress_{YEAR}.csv"

SEARCH_TERMS = {
    "AAPL.NAS": '("Apple Inc" OR AAPL)',
    "MSFT.NAS": '(Microsoft OR MSFT)',
    "NVDA.NAS": '(NVIDIA OR NVDA)',
    "AMZN.NAS": '("Amazon Inc" OR AMZN)',
    "GOOG.NAS": '("Alphabet Inc" OR "Google LLC" OR GOOG)',
    "NFLX.NAS": '(Netflix OR NFLX)',
    "AMD.NAS": '("Advanced Micro Devices" OR AMD)',
    "TSLA.NAS": '(Tesla OR TSLA)',
    "NDAQ.NAS": '("Nasdaq Inc" OR NDAQ)',
    "SBUX.NAS": '("Starbucks Corporation" OR Starbucks OR SBUX)',
    "ADBE.NAS": '("Adobe Inc" OR ADBE)',
    "MVRS.NAS": '("Meta Platforms" OR "Meta Platforms Inc" OR Facebook)',
    "NKE.NYSE": '(Nike OR NKE)',
    "CRM.NYSE": '(Salesforce OR CRM)',
    "PYPL.NAS": '(PayPal OR PYPL)',
}


def parse_gdelt_seendate(seendate: str) -> datetime | None:
    """
    Convert GDELT seendate like:
        20260318T154500Z
    into a timezone-aware UTC datetime.
    """
    if not seendate:
        return None

    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def ensure_csv_headers() -> None:
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=[
            "symbol",
            "title",
            "url",
            "domain",
            "seendate",
            "seen_at_utc",
            "language",
            "sourcecountry",
        ]).to_csv(OUTPUT_FILE, index=False)

    if not os.path.exists(PROGRESS_FILE):
        pd.DataFrame(columns=[
            "symbol",
            "date",
            "status",
            "article_count",
            "logged_at_utc",
        ]).to_csv(PROGRESS_FILE, index=False)


def load_completed_days() -> set[tuple[str, str]]:
    if not os.path.exists(PROGRESS_FILE):
        return set()

    try:
        progress_df = pd.read_csv(PROGRESS_FILE)
        if progress_df.empty:
            return set()

        handled_rows = progress_df[
            progress_df["status"].isin(["done", "json_fixed", "json_failed"])
        ]
        return set(zip(handled_rows["symbol"], handled_rows["date"]))
    except Exception as e:
        print(f"[WARN] Could not read progress file cleanly: {e}")
        return set()


def append_articles(rows: list[dict]) -> None:
    if not rows:
        return

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)


def append_progress(symbol: str, date_str: str, status: str, article_count: int) -> None:
    pd.DataFrame([{
        "symbol": symbol,
        "date": date_str,
        "status": status,
        "article_count": article_count,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
    }]).to_csv(PROGRESS_FILE, mode="a", header=False, index=False)


def repair_invalid_json_escapes(text: str) -> str:
    # Replace backslashes that are NOT followed by a valid JSON escape character
    # valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def parse_gdelt_json_response(response_text: str) -> tuple[dict, str]:
    try:
        data = json.loads(response_text)
        return data, "done"
    except json.JSONDecodeError as e1:
        repaired = repair_invalid_json_escapes(response_text)
        try:
            data = json.loads(repaired)
            return data, "json_fixed"
        except json.JSONDecodeError as e2:
            raise ValueError(f"Original JSON error: {e1}; repaired JSON error: {e2}")


def fetch_day_until_handled(
    session: requests.Session,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    symbol: str,
) -> tuple[list[dict], int, str]:
    params = {
        "query": f"{query} AND sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }

    attempt = 0
    json_error_count = 0

    while True:
        attempt += 1

        try:
            response = session.get(BASE_URL, params=params, timeout=30)

            if response.status_code != 200:
                print(
                    f"[WARN] {symbol} {start_dt.date()} "
                    f"status={response.status_code} attempt={attempt}. "
                    f"Sleeping {RETRY_SLEEP}s before retry..."
                )
                time.sleep(RETRY_SLEEP)
                continue

            try:
                data, parse_status = parse_gdelt_json_response(response.text)
                return data.get("articles", []), attempt, parse_status

            except ValueError:
                json_error_count += 1
                print(
                    f"[ERROR] {symbol} {start_dt.date()} "
                    f"attempt={attempt} malformed JSON "
                    f"({json_error_count}/{MAX_JSON_ERRORS_PER_DAY})"
                )

                if json_error_count >= MAX_JSON_ERRORS_PER_DAY:
                    print(
                        f"[FAIL] {symbol} {start_dt.date()} could not be parsed after "
                        f"{MAX_JSON_ERRORS_PER_DAY} malformed JSON attempts. "
                        f"Marking as json_failed and continuing."
                    )
                    return [], attempt, "json_failed"

                time.sleep(RETRY_SLEEP)
                continue

        except requests.RequestException as e:
            print(
                f"[ERROR] {symbol} {start_dt.date()} "
                f"attempt={attempt} request error={e}. "
                f"Sleeping {RETRY_SLEEP}s before retry..."
            )
            time.sleep(RETRY_SLEEP)


def clean_articles(articles: list[dict], symbol: str) -> list[dict]:
    cleaned = []
    local_seen_urls = set()

    for article in articles:
        url = article.get("url")
        seendate = article.get("seendate")

        if not url or url in local_seen_urls:
            continue
        local_seen_urls.add(url)

        seen_dt = parse_gdelt_seendate(seendate)
        seen_at_utc = seen_dt.isoformat() if seen_dt is not None else None

        cleaned.append({
            "symbol": symbol,
            "title": article.get("title"),
            "url": url,
            "domain": article.get("domain"),
            "seendate": seendate,
            "seen_at_utc": seen_at_utc,
            "language": article.get("language"),
            "sourcecountry": article.get("sourcecountry"),
        })

    return cleaned


def run_collection() -> None:
    ensure_csv_headers()
    completed_days = load_completed_days()

    total_start_time = time.time()
    total_api_calls = 0
    skipped_days = 0
    completed_day_count = 0

    session = requests.Session()
    session.headers.update({
        "User-Agent": "thesis-gdelt-fetcher/1.0",
        "Accept": "application/json",
    })

    total_symbols = len(SEARCH_TERMS)

    for symbol_index, (symbol, query) in enumerate(SEARCH_TERMS.items(), start=1):
        symbol_start_time = time.time()
        symbol_api_calls = 0
        symbol_days_done = 0
        symbol_days_skipped = 0

        print(f"\n{'=' * 80}")
        print(f"Starting symbol {symbol_index}/{total_symbols}: {symbol}")
        print(f"{'=' * 80}")

        current_day = START_DATE

        while current_day < END_DATE:
            day_start = current_day
            day_end = current_day + timedelta(days=1)
            date_str = day_start.strftime("%Y-%m-%d")
            key = (symbol, date_str)

            if key in completed_days:
                symbol_days_skipped += 1
                skipped_days += 1
                print(f"[SKIP] symbol={symbol} date={date_str} already handled")
                current_day = day_end
                continue

            call_start = time.time()
            articles, attempts_used, fetch_status = fetch_day_until_handled(
                session=session,
                query=query,
                start_dt=day_start,
                end_dt=day_end,
                symbol=symbol,
            )
            call_time = round(time.time() - call_start, 2)

            total_api_calls += attempts_used
            symbol_api_calls += attempts_used

            cleaned_articles = clean_articles(articles, symbol)
            append_articles(cleaned_articles)
            append_progress(symbol, date_str, fetch_status, len(cleaned_articles))

            completed_days.add(key)
            completed_day_count += 1
            symbol_days_done += 1

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(
                f"[{ts}] "
                f"symbol={symbol} "
                f"date={date_str} "
                f"status={fetch_status} "
                f"articles={len(cleaned_articles)} "
                f"call_time={call_time}s"
            )

            time.sleep(RATE_LIMIT)
            current_day = day_end

        symbol_runtime = round(time.time() - symbol_start_time, 2)
        print(
            f"Finished {symbol}: "
            f"days_done={symbol_days_done}, "
            f"days_skipped={symbol_days_skipped}, "
            f"api_calls={symbol_api_calls}, "
            f"runtime={symbol_runtime}s"
        )

    total_runtime = round(time.time() - total_start_time, 2)

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"Total API calls this run: {total_api_calls}")
    print(f"Completed/handled days this run: {completed_day_count}")
    print(f"Skipped already-handled days: {skipped_days}")
    print(f"Total runtime: {total_runtime} seconds")
    print(f"Total runtime: {round(total_runtime / 60, 2)} minutes")
    print(f"Articles file: {OUTPUT_FILE}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_collection()
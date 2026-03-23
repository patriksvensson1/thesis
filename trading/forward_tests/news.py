import os
import time
import json
import re
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta, timezone

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_NEWS_LOG_FILE = DATA_DIR / "gdelt_news_log.csv"


def parse_gdelt_seendate(seendate: str) -> datetime:
    # Example: "20260318T154500Z"
    return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def parse_seen_at_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def rebuild_seen_urls(article_store: dict[str, list[dict]], symbols: list[str]) -> dict[str, set[str]]:
    rebuilt = {symbol: set() for symbol in symbols}

    for symbol in symbols:
        for article in article_store.get(symbol, []):
            url = article.get("url")
            if url:
                rebuilt[symbol].add(url)

    return rebuilt


def repair_invalid_json_escapes(text: str) -> str:
    return re.sub(
        r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
        r'\\\\',
        text
    )


def parse_gdelt_json_response(response_text: str) -> tuple[dict, str]:
    try:
        return json.loads(response_text), "done"
    except json.JSONDecodeError as e1:
        repaired = repair_invalid_json_escapes(response_text)
        try:
            return json.loads(repaired), "json_fixed"
        except json.JSONDecodeError as e2:
            raise ValueError(f"Original JSON error: {e1}; repaired JSON error: {e2}")


def ensure_news_log_file(news_log_file: Path) -> None:
    news_log_file.parent.mkdir(parents=True, exist_ok=True)

    if not news_log_file.exists():
        pd.DataFrame(columns=[
            "symbol",
            "title",
            "url",
            "source",
            "seendate",
            "seen_at_utc",
            "language",
            "sourcecountry",
            "fetched_at_utc",
        ]).to_csv(news_log_file, index=False)


def append_news_log(news_log_file: Path, rows: list[dict]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(news_log_file, mode="a", header=False, index=False)


def get_gdelt_json_with_retry(
    session: requests.Session,
    base_url: str,
    params: dict,
    symbol: str,
    max_retries: int = 3,
    min_wait_seconds: float = 5.0,
    timeout: int = 30,
) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] Requesting GDELT for {symbol} (attempt {attempt}/{max_retries})")

            response = session.get(base_url, params=params, timeout=timeout)

            content_type = response.headers.get("Content-Type", "")
            body_preview = response.text[:200].replace("\n", " ").replace("\r", " ")

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else min_wait_seconds

                if attempt < max_retries:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] GDELT rate limit hit for {symbol} "
                        f"(attempt {attempt}/{max_retries}). "
                        f"Sleeping {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                    continue
                else:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] GDELT rate limit hit for {symbol} "
                        f"after {max_retries} attempts."
                    )
                    return {"articles": []}

            response.raise_for_status()

            if not response.text.strip():
                if attempt < max_retries:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Empty response from GDELT for {symbol} "
                        f"(attempt {attempt}/{max_retries}). "
                        f"Sleeping {min_wait_seconds} seconds..."
                    )
                    time.sleep(min_wait_seconds)
                    continue
                else:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Empty response from GDELT for {symbol} "
                        f"after {max_retries} attempts."
                    )
                    return {"articles": []}

            if "json" not in content_type.lower():
                if attempt < max_retries:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Non-JSON response from GDELT for {symbol} "
                        f"(Content-Type: {content_type}). "
                        f"Body preview: {body_preview!r}. "
                        f"Sleeping {min_wait_seconds} seconds..."
                    )
                    time.sleep(min_wait_seconds)
                    continue
                else:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Non-JSON response from GDELT for {symbol} "
                        f"after {max_retries} attempts. "
                        f"Body preview: {body_preview!r}"
                    )
                    return {"articles": []}

            try:
                data, parse_status = parse_gdelt_json_response(response.text)

                if parse_status == "json_fixed":
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Repaired malformed JSON from GDELT for {symbol} "
                        f"on attempt {attempt}/{max_retries}."
                    )

                return data

            except ValueError as e:
                if attempt < max_retries:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Invalid JSON from GDELT for {symbol} "
                        f"(attempt {attempt}/{max_retries}): {e}. "
                        f"Body preview: {body_preview!r}. "
                        f"Sleeping {min_wait_seconds} seconds..."
                    )
                    time.sleep(min_wait_seconds)
                    continue
                else:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{ts}] Invalid JSON from GDELT for {symbol} "
                        f"after {max_retries} attempts: {e}. "
                        f"Body preview: {body_preview!r}"
                    )
                    return {"articles": []}

        except requests.RequestException as e:
            if attempt < max_retries:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{ts}] GDELT request failed for {symbol} "
                    f"(attempt {attempt}/{max_retries}): {e}. "
                    f"Sleeping {min_wait_seconds} seconds..."
                )
                time.sleep(min_wait_seconds)
            else:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{ts}] GDELT request failed for {symbol} "
                    f"after {max_retries} attempts: {e}"
                )
                return {"articles": []}

    return {"articles": []}


def fetch_gdelt_news(
    symbols: list[str],
    max_records_per_symbol: int = 50,
    timespan: str = "24h",
    seen_urls: Optional[dict[str, set[str]]] = None,
    article_store: Optional[dict[str, list[dict]]] = None,
    active_window_hours: int = 24,
    max_retries: int = 6,
    min_wait_seconds: float = 8.0,
    news_log_file: Path = DEFAULT_NEWS_LOG_FILE,
) -> dict[str, list[dict]]:
    if seen_urls is None:
        seen_urls = {symbol: set() for symbol in symbols}

    if article_store is None:
        article_store = {symbol: [] for symbol in symbols}

    ensure_news_log_file(news_log_file)

    search_terms = {
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

    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=active_window_hours)
    fetched_at_utc = now_utc.isoformat()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "thesis-news-fetcher/1.0",
            "Accept": "application/json",
        }
    )

    rows_to_log = []

    for symbol in symbols:
        if symbol not in article_store:
            article_store[symbol] = []

        if symbol not in seen_urls:
            seen_urls[symbol] = set()

        base_query = search_terms.get(symbol, f'({symbol.split(".")[0]})')
        query = f"{base_query} AND sourcelang:english"

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": max_records_per_symbol,
            "timespan": timespan,
        }

        data = get_gdelt_json_with_retry(
            session=session,
            base_url=base_url,
            params=params,
            symbol=symbol,
            max_retries=max_retries,
            min_wait_seconds=min_wait_seconds,
        )

        articles = data.get("articles", [])
        local_seen = set()

        for article in articles:
            url = article.get("url")
            seendate = article.get("seendate")

            if not url or not seendate:
                continue

            if url in local_seen:
                continue
            local_seen.add(url)

            if url in seen_urls[symbol]:
                continue

            try:
                article_time = parse_gdelt_seendate(seendate)
            except ValueError:
                continue

            if article_time < cutoff:
                continue

            cleaned_article = {
                "symbol": symbol,
                "title": article.get("title"),
                "url": url,
                "source": article.get("domain"),
                "seendate": seendate,
                "seen_at_utc": article_time.isoformat(),
                "language": article.get("language"),
                "sourcecountry": article.get("sourcecountry"),
            }

            article_store[symbol].append(cleaned_article)
            seen_urls[symbol].add(url)

            rows_to_log.append({
                **cleaned_article,
                "fetched_at_utc": fetched_at_utc,
            })

        pruned_articles = []
        for article in article_store[symbol]:
            seen_at_utc = article.get("seen_at_utc")
            if seen_at_utc is None:
                continue

            try:
                article_seen_time = parse_seen_at_utc(seen_at_utc)
            except ValueError:
                continue

            if article_seen_time >= cutoff:
                pruned_articles.append(article)

        article_store[symbol] = pruned_articles

        article_store[symbol].sort(
            key=lambda x: parse_seen_at_utc(x["seen_at_utc"]),
            reverse=True,
        )

        time.sleep(min_wait_seconds)

    append_news_log(news_log_file, rows_to_log)

    rebuilt_seen_urls = rebuild_seen_urls(article_store, symbols)
    seen_urls.clear()
    seen_urls.update(rebuilt_seen_urls)

    return article_store
import time
import requests
from typing import Optional
from datetime import datetime, timedelta, timezone


def parse_gdelt_seendate(seendate: str) -> datetime:
    # Example: "20260318T154500Z"
    return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def get_gdelt_json_with_retry(
    session: requests.Session,
    base_url: str,
    params: dict,
    symbol: str,
    max_retries: int = 3,
    min_wait_seconds: float = 5.0,
    timeout: int = 30,
) -> dict:
    """
    Try the GDELT request up to max_retries times.
    Wait between attempts.
    Handle:
    - 429 rate limits
    - empty bodies
    - non-JSON responses
    - invalid JSON
    """

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
                return response.json()
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
    seen_urls: Optional[set[str]] = None,
    article_store: Optional[dict[str, list[dict]]] = None,
    active_window_hours: int = 24,
    max_retries: int = 6,
    min_wait_seconds: float = 9.0,
) -> dict[str, list[dict]]:
    """
    Fetch recent GDELT news, deduplicate by URL, and maintain a rolling
    per-symbol article store that can later be used for decay.
    """

    if seen_urls is None:
        seen_urls = set()

    if article_store is None:
        article_store = {symbol: [] for symbol in symbols}

    search_terms = {
        "AAPL.US": '("Apple Inc" OR AAPL)',
        "MSFT.US": '(Microsoft OR MSFT)',
        "NVDA.US": '(NVIDIA OR NVDA)',
        "AMZN.US": '("Amazon Inc" OR AMZN)',
        "GOOGL.US": '(Google OR Alphabet OR GOOGL)',
        "NFLX.US": '(Netflix OR NFLX)',
        "WBD.US": '("Warner Bros Discovery" OR WBD)',
        "TSLA.US": '(Tesla OR TSLA)',
        "NDAQ.US": '(Nasdaq OR NDAQ)',
        "SBUX.US": '("Starbucks Corporation" OR Starbucks OR SBUX)',
        "ADBE.US": '(Adobe OR ADBE)',
        "META.US": '("Meta Platforms" OR "Meta Platforms Inc" OR Facebook)',
        "NKE.US": '(Nike OR NKE)',
        "CRM.US": '(Salesforce OR CRM)',
        "PYPL.US": '(PayPal OR PYPL)',
    }

    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=active_window_hours)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "thesis-news-fetcher/1.0",
            "Accept": "application/json",
        }
    )

    for symbol in symbols:
        base_query = search_terms.get(symbol, f'({symbol.split(".")[0]})')

        # Restrict to English-language source articles
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

        if symbol not in article_store:
            article_store[symbol] = []

        for article in articles:
            url = article.get("url")
            seendate = article.get("seendate")

            if not url or not seendate:
                continue

            if url in local_seen:
                continue
            local_seen.add(url)

            if url in seen_urls:
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
                "seen_at_utc": article_time,
                "language": article.get("language"),
                "sourcecountry": article.get("sourcecountry"),
            }

            article_store[symbol].append(cleaned_article)
            seen_urls.add(url)

        article_store[symbol] = [
            article
            for article in article_store[symbol]
            if article.get("seen_at_utc") is not None and article["seen_at_utc"] >= cutoff
        ]

        article_store[symbol].sort(
            key=lambda x: x["seen_at_utc"],
            reverse=True,
        )

        time.sleep(min_wait_seconds)

    return article_store
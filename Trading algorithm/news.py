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
    Wait at least min_wait_seconds between attempts.
    If GDELT responds with 429 and a Retry-After header, use that instead.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(base_url, params=params, timeout=timeout)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else min_wait_seconds
                print(
                    f"GDELT rate limit hit for {symbol} "
                    f"(attempt {attempt}/{max_retries}). "
                    f"Sleeping {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            last_error = e

            if attempt < max_retries:
                print(
                    f"GDELT request failed for {symbol} "
                    f"(attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {min_wait_seconds} seconds..."
                )
                time.sleep(min_wait_seconds)
            else:
                print(
                    f"GDELT request failed for {symbol} "
                    f"after {max_retries} attempts: {e}"
                )

        except ValueError as e:
            last_error = e

            if attempt < max_retries:
                print(
                    f"Invalid JSON from GDELT for {symbol} "
                    f"(attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {min_wait_seconds} seconds..."
                )
                time.sleep(min_wait_seconds)
            else:
                print(
                    f"Invalid JSON from GDELT for {symbol} "
                    f"after {max_retries} attempts: {e}"
                )

    return {"articles": []}


def fetch_gdelt_news(
    symbols: list[str],
    max_records_per_symbol: int = 50,
    timespan: str = "24h",
    seen_urls: Optional[set[str]] = None,
    article_store: Optional[dict[str, list[dict]]] = None,
    active_window_hours: int = 24,
    max_retries: int = 6,
    min_wait_seconds: float = 6.0,
) -> dict[str, list[dict]]:
    """
    Fetch recent GDELT news, deduplicate by URL, and maintain a rolling
    per-symbol article store that can later be used for decay.

    seen_urls:
        URLs already ingested before, used only for deduplication.

    article_store:
        Rolling memory of recent articles per symbol.
        This is what decay logic should use later, not seen_urls.

    Returns:
        article_store filtered to the active time window.
    """

    if seen_urls is None:
        seen_urls = set()

    if article_store is None:
        article_store = {symbol: [] for symbol in symbols}

    search_terms = {
        "AAPL.US": '"Apple Inc" OR AAPL',
        "MSFT.US": '"Microsoft" OR MSFT',
        "NVDA.US": '"NVIDIA" OR NVDA',
        "AMZN.US": '"Amazon" OR AMZN',
        "GOOGL.US": '"Google" OR Alphabet OR GOOGL',
        "NFLX.US": '"Netflix" OR NFLX',
        "WBD.US": '"Warner Bros Discovery" OR WBD',
        "TSLA.US": '"Tesla" OR TSLA',
        "NDAQ.US": '"Nasdaq" OR NDAQ',
        "SBUX.US": '"Starbucks Corporation" OR SBUS',
        "ADBE.US": '"Adobe" OR ADBE',
        "META.US": '"Meta Platforms" OR "MMeta Platforms Technologies" OR "Facebook"',
        "NKE.US": '"Nike" OR NKE',
        "CRM.US": '"Salesforce" OR CRM',
        "PYPL.US": '"PayPal" OR PYPL',
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
        query = search_terms.get(symbol, symbol.split(".")[0])

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

            # Never keep articles older than the active decay window
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

        # Prune old articles so decay later only sees recent window
        article_store[symbol] = [
            article
            for article in article_store[symbol]
            if article.get("seen_at_utc") is not None and article["seen_at_utc"] >= cutoff
        ]

        # Keep newest first
        article_store[symbol].sort(
            key=lambda x: x["seen_at_utc"],
            reverse=True,
        )

        # Sleep between symbols to avoid hammering the API
        time.sleep(min_wait_seconds)

    return article_store
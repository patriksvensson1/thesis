from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_sentiment_log_file(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        pd.DataFrame(columns=[
            "symbol",
            "title",
            "url",
            "source",
            "seendate",
            "seen_at_utc",
            "language",
            "sourcecountry",
            "sentiment_label",
            "sentiment_confidence",
            "sentiment_score",
            "logged_at_utc",
        ]).to_csv(log_file, index=False)


def load_logged_sentiment_urls(log_file: Path) -> set[str]:
    ensure_sentiment_log_file(log_file)

    try:
        df = pd.read_csv(log_file, usecols=["url"])
        return set(df["url"].dropna().astype(str))
    except Exception:
        return set()


def append_sentiment_log(
    scored_news_by_symbol: dict[str, list[dict]],
    logged_urls: set[str],
    logged_at_utc: str,
    log_file: Path,
) -> int:
    ensure_sentiment_log_file(log_file)

    rows_to_append: list[dict] = []

    for symbol, articles in scored_news_by_symbol.items():
        for article in articles:
            url = article.get("url")
            if not url:
                continue

            url = str(url)
            if url in logged_urls:
                continue

            rows_to_append.append({
                "symbol": symbol,
                "title": article.get("title"),
                "url": url,
                "source": article.get("source"),
                "seendate": article.get("seendate"),
                "seen_at_utc": article.get("seen_at_utc"),
                "language": article.get("language"),
                "sourcecountry": article.get("sourcecountry"),
                "sentiment_label": article.get("sentiment_label"),
                "sentiment_confidence": article.get("sentiment_confidence"),
                "sentiment_score": article.get("sentiment_score"),
                "logged_at_utc": logged_at_utc,
            })

            logged_urls.add(url)

    if not rows_to_append:
        return 0

    pd.DataFrame(rows_to_append).to_csv(
        log_file,
        mode="a",
        header=False,
        index=False,
    )
    return len(rows_to_append)
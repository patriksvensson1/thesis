from copy import deepcopy
from transformers import pipeline

_finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)

_SENTIMENT_CACHE: dict[str, tuple[str, float, float]] = {}


def _label_to_score(label: str, score: float) -> float:
    label = label.lower()

    if label == "positive":
        return float(score)
    if label == "negative":
        return -float(score)
    return 0.0


def _get_headline_sentiment(headline: str) -> tuple[str, float, float]:
    if not headline or not headline.strip():
        return "neutral", 0.0, 0.0

    headline = headline.strip()

    cached = _SENTIMENT_CACHE.get(headline)
    if cached is not None:
        return cached

    result = _finbert(headline, truncation=True)[0]
    label = result["label"].lower()
    confidence = float(result["score"])
    sentiment_score = _label_to_score(label, confidence)

    output = (label, confidence, sentiment_score)
    _SENTIMENT_CACHE[headline] = output
    return output


def compute_sentiment_scores(news_by_symbol: dict[str, list[dict]]) -> dict[str, list[dict]]:
    if not news_by_symbol:
        return {}

    scored_news = deepcopy(news_by_symbol)

    for symbol, articles in scored_news.items():
        if not articles:
            scored_news[symbol] = []
            continue

        for article in articles:
            title = str(article.get("title", "")).strip()
            label, confidence, sentiment_score = _get_headline_sentiment(title)

            article["sentiment_label"] = label
            article["sentiment_confidence"] = confidence
            article["sentiment_score"] = sentiment_score

    return scored_news
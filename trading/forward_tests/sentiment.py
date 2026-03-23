from copy import deepcopy
from transformers import pipeline

# Load FinBERT once when this file is imported.
_finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)


def _label_to_score(label: str, score: float) -> float:
    """
    Convert FinBERT output into one numeric sentiment score.

    positive -> +score
    negative -> -score
    neutral  -> 0.0
    """
    label = label.lower()

    if label == "positive":
        return float(score)
    if label == "negative":
        return -float(score)
    return 0.0


def _get_headline_sentiment(headline: str) -> tuple[str, float, float]:
    """
    Score one headline and return:
        (label, confidence, numeric_score)

    Example:
        ("positive", 0.93, 0.93)
        ("negative", 0.88, -0.88)
        ("neutral", 0.76, 0.0)
    """
    if not headline or not headline.strip():
        return "neutral", 0.0, 0.0

    result = _finbert(headline, truncation=True)[0]
    label = result["label"].lower()
    confidence = float(result["score"])
    sentiment_score = _label_to_score(label, confidence)

    return label, confidence, sentiment_score


def compute_sentiment_scores(news_by_symbol: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """
    Add sentiment info to each article.

    Input:
        {
            "AAPL.US": [article_dict, ...],
            "MSFT.US": [],
            ...
        }

    Output:
        Same structure, but each article gets:
            sentiment_label
            sentiment_confidence
            sentiment_score

    Notes:
    - If a symbol has no articles, it stays an empty list.
    - If GDELT failed for a symbol earlier and returned no articles,
      this function simply leaves that symbol empty.
    """
    if not news_by_symbol:
        return {}

    scored_news = deepcopy(news_by_symbol)

    for symbol, articles in scored_news.items():
        # If no articles for this symbol, keep it as an empty list
        if not articles:
            scored_news[symbol] = []
            continue

        for article in articles:
            title = article.get("title", "").strip()
            label, confidence, sentiment_score = _get_headline_sentiment(title)

            article["sentiment_label"] = label
            article["sentiment_confidence"] = confidence
            article["sentiment_score"] = sentiment_score

    return scored_news
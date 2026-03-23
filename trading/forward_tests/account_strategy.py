from datetime import datetime, timezone

LSTM_WEIGHT = 0.5
NEWS_WEIGHT = 0.5

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15

DECAY_WINDOW_HOURS = 24


def _get_now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_article_time(article: dict) -> datetime | None:
    article_time = article.get("seen_at_utc")

    if article_time is None:
        return None

    if isinstance(article_time, str):
        try:
            article_time = datetime.fromisoformat(article_time)
        except ValueError:
            return None

    if article_time.tzinfo is None:
        return article_time.replace(tzinfo=timezone.utc)

    return article_time.astimezone(timezone.utc)


def _article_sentiment(article: dict) -> float:
    score = article.get("sentiment_score")
    if score is None:
        return 0.0
    return float(score)


def _average_news_score_no_decay(scored_articles: list[dict]) -> float:
    if not scored_articles:
        return 0.0

    scores = [_article_sentiment(article) for article in scored_articles]
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _average_news_score_with_decay(
    scored_articles: list[dict],
    now_utc: datetime,
) -> float:
    """
    Simple linear decay over 24 hours.

    Weight formula:
        weight = 1 - (age_hours / DECAY_WINDOW_HOURS)

    Examples:
        0 hours old  -> weight 1.0
        6 hours old  -> weight 0.75
        12 hours old -> weight 0.5
        24 hours old -> weight 0.0
    """
    if not scored_articles:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0

    for article in scored_articles:
        article_time = _safe_article_time(article)
        sentiment_score = _article_sentiment(article)

        if article_time is None:
            continue

        age_seconds = (now_utc - article_time).total_seconds()
        age_hours = max(age_seconds / 3600.0, 0.0)

        if age_hours >= DECAY_WINDOW_HOURS:
            weight = 0.0
        else:
            weight = 1.0 - (age_hours / DECAY_WINDOW_HOURS)

        weighted_sum += sentiment_score * weight
        weight_total += weight

    if weight_total == 0.0:
        return 0.0

    return weighted_sum / weight_total


def _get_news_score_for_account(
    account: dict,
    scored_articles: list[dict],
    now_utc: datetime,
) -> float:
    account_name = account.get("name", "").lower()

    if "decay" in account_name and "no_decay" not in account_name:
        return _average_news_score_with_decay(scored_articles, now_utc)

    return _average_news_score_no_decay(scored_articles)


def _get_lstm_score(lstm_prediction: dict | None) -> float:
    if not lstm_prediction:
        return 0.0

    prob_up = lstm_prediction.get("prob_up")
    if prob_up is None:
        return 0.0

    prob_up = float(prob_up)
    return 2.0 * prob_up - 1.0


def _combine_scores(news_score: float, lstm_score: float) -> float:
    return NEWS_WEIGHT * news_score + LSTM_WEIGHT * lstm_score


def _score_to_action(final_score: float) -> str:
    if final_score >= BUY_THRESHOLD:
        return "buy"
    if final_score <= SELL_THRESHOLD:
        return "sell"
    return "hold"


def apply_account_decay_and_rank(
    symbols: list[str],
    news_by_symbol: dict[str, list[dict]],
    sentiment_scores: dict[str, list[dict]],
    lstm_predictions: dict[str, dict],
    account: dict,
) -> list[dict]:
    now_utc = _get_now_utc()
    ranked_opportunities = []

    for symbol in symbols:
        scored_articles = sentiment_scores.get(symbol, [])
        lstm_prediction = lstm_predictions.get(symbol, {})

        news_score = _get_news_score_for_account(
            account=account,
            scored_articles=scored_articles,
            now_utc=now_utc,
        )

        lstm_score = _get_lstm_score(lstm_prediction)
        final_score = _combine_scores(news_score, lstm_score)
        action = _score_to_action(final_score)

        ranked_opportunities.append(
            {
                "symbol": symbol,
                "account_name": account.get("name"),
                "news_score": news_score,
                "lstm_score": lstm_score,
                "final_score": final_score,
                "action": action,
                "prob_up": lstm_prediction.get("prob_up"),
                "predicted_class": lstm_prediction.get("predicted_class"),
                "article_count": len(scored_articles),
            }
        )

    ranked_opportunities.sort(
        key=lambda item: abs(item["final_score"]),
        reverse=True,
    )

    return ranked_opportunities
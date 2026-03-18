import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import SYMBOLS, ACCOUNTS, CHECK_EVERY_MINUTES, MAX_HOLD_MINUTES
from mt5_utils import (
    initialize_mt5,
    login_to_account,
    get_open_positions,
    close_position,
)
from news import fetch_gdelt_news
from sentiment import compute_sentiment_scores
from lstm_model import get_lstm_predictions
from account_strategy import apply_account_decay_and_rank
from trader import execute_best_trade

# Strategy run window
EASTERN_TZ = ZoneInfo("America/New_York")
START_HOUR = 9
START_MINUTE = 30
END_HOUR = 16
END_MINUTE = 00


def should_run_now(last_run: datetime | None, interval_minutes: int) -> bool:
    now_et = datetime.now(EASTERN_TZ)

    market_open = now_et.replace(
        hour=START_HOUR,
        minute=START_MINUTE,
        second=0,
        microsecond=0,
    )
    market_close = now_et.replace(
        hour=END_HOUR,
        minute=END_MINUTE,
        second=0,
        microsecond=0,
    )

    if now_et < market_open or now_et > market_close:
        return False

    if last_run is None:
        return True

    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=EASTERN_TZ)
    else:
        last_run = last_run.astimezone(EASTERN_TZ)

    return now_et - last_run >= timedelta(minutes=interval_minutes)


def close_expired_positions_for_account():
    open_positions = get_open_positions()
    now = datetime.now()

    for pos in open_positions:
        opened_at = datetime.fromtimestamp(pos.time)
        age = now - opened_at

        if age >= timedelta(minutes=MAX_HOLD_MINUTES):
            close_position(pos)


def main():
    initialize_mt5()
    last_run = None

    seen_urls = set()
    article_store = {symbol: [] for symbol in SYMBOLS}

    while True:
        if not should_run_now(last_run, CHECK_EVERY_MINUTES):
            time.sleep(1)
            continue

        print(f"\nRunning cycle at {datetime.now()}")

        # --------------------------------------------------
        # 1. Get shared data once for all accounts
        # --------------------------------------------------
        news_by_symbol = fetch_gdelt_news(
            symbols=SYMBOLS,
            seen_urls=seen_urls,
            article_store=article_store,
            active_window_hours=24,
        )

        lstm_predictions = get_lstm_predictions(SYMBOLS)
        sentiment_scores = compute_sentiment_scores(news_by_symbol)

        # --------------------------------------------------
        # 2. Run account-specific strategy logic
        # --------------------------------------------------
        for account in ACCOUNTS:
            login_to_account(account)

            # Close positions that have been open too long
            close_expired_positions_for_account()

            # Apply this account's decay / scoring logic
            ranked_opportunities = apply_account_decay_and_rank(
                symbols=SYMBOLS,
                news_by_symbol=news_by_symbol,
                sentiment_scores=sentiment_scores,
                lstm_predictions=lstm_predictions,
                account=account,
            )

            # Buy / sell best candidate if strategy says so
            execute_best_trade(account, ranked_opportunities)

        last_run = datetime.now()


if __name__ == "__main__":
    main()
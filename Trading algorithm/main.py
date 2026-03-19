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

# Strategy run window in U.S. market time
EASTERN_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

START_HOUR = 9
START_MINUTE = 30
END_HOUR = 16
END_MINUTE = 0


def get_market_open_close(now_et: datetime) -> tuple[datetime, datetime]:
    """
    Return today's market open and close in Eastern Time.
    """
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
    return market_open, market_close


def get_next_run_time(interval_minutes: int) -> datetime:
    """
    Schedule the next run on fixed 15-minute boundaries inside the market window.

    Examples for interval=15:
        09:30, 09:45, 10:00, 10:15, ...

    If current time is before market open, return today's open.
    If current time is after market close, return next day's open.
    """
    now_et = datetime.now(EASTERN_TZ)
    market_open, market_close = get_market_open_close(now_et)

    if now_et < market_open:
        return market_open

    if now_et >= market_close:
        next_day = now_et + timedelta(days=1)
        return next_day.replace(
            hour=START_HOUR,
            minute=START_MINUTE,
            second=0,
            microsecond=0,
        )

    minutes_since_open = int((now_et - market_open).total_seconds() // 60)
    next_slot_minutes = ((minutes_since_open // interval_minutes) + 1) * interval_minutes
    next_run = market_open + timedelta(minutes=next_slot_minutes)

    if next_run > market_close:
        next_day = now_et + timedelta(days=1)
        return next_day.replace(
            hour=START_HOUR,
            minute=START_MINUTE,
            second=0,
            microsecond=0,
        )

    return next_run


def sleep_until_next_run(interval_minutes: int) -> None:
    """
    Sleep until the next scheduled run time.
    """
    next_run = get_next_run_time(interval_minutes)
    now_et = datetime.now(EASTERN_TZ)
    sleep_seconds = (next_run - now_et).total_seconds()

    if sleep_seconds > 0:
        print(f"Sleeping until next cycle at {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        time.sleep(sleep_seconds)


def close_expired_positions_for_account():
    """
    Close positions in the currently logged-in MT5 account if they have been
    open longer than MAX_HOLD_MINUTES.
    """
    open_positions = get_open_positions()

    if not open_positions:
        return

    now_utc = datetime.now(UTC_TZ)
    max_age = timedelta(minutes=MAX_HOLD_MINUTES)

    for pos in open_positions:
        try:
            opened_at_utc = datetime.fromtimestamp(pos.time, tz=UTC_TZ)
            age = now_utc - opened_at_utc

            if age >= max_age:
                print(
                    f"Closing expired position | "
                    f"symbol={pos.symbol}, ticket={pos.ticket}, age={age}"
                )
                close_position(pos)

        except Exception as e:
            print(
                f"Failed to process/close position "
                f"{getattr(pos, 'ticket', 'unknown')}: {e}"
            )


def run_one_cycle(seen_urls: set[str], article_store: dict[str, list[dict]]) -> None:
    """
    Run one full strategy cycle.
    """
    print(f"\nRunning cycle at {datetime.now()}")

    # Log into one account first so MT5 market data is available
    login_to_account(ACCOUNTS[0])

    # --------------------------------------------------
    # 1. Get shared data once for all accounts
    # --------------------------------------------------
    news_by_symbol = fetch_gdelt_news(
        symbols=SYMBOLS,
        seen_urls=seen_urls,
        article_store=article_store,
        active_window_hours=24,
    )

    sentiment_scores = compute_sentiment_scores(news_by_symbol)
    lstm_predictions = get_lstm_predictions(SYMBOLS)

    # --------------------------------------------------
    # 2. Run account-specific strategy logic
    # --------------------------------------------------
    for account in ACCOUNTS:
        login_to_account(account)

        # Uncomment when you want time-based closing active
        # close_expired_positions_for_account()

        ranked_opportunities = apply_account_decay_and_rank(
            symbols=SYMBOLS,
            news_by_symbol=news_by_symbol,
            sentiment_scores=sentiment_scores,
            lstm_predictions=lstm_predictions,
            account=account,
        )

        # Uncomment when you want live trade execution active
        # execute_best_trade(account, ranked_opportunities)

        print(account["name"], ranked_opportunities[:3])


def main():
    initialize_mt5()

    seen_urls = set()
    article_store = {symbol: [] for symbol in SYMBOLS}

    while True:
        # sleep_until_next_run(CHECK_EVERY_MINUTES)
        run_one_cycle(seen_urls, article_store)


if __name__ == "__main__":
    main()
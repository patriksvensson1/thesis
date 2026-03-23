import json
import os
import time
from pathlib import Path
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
from sentiment_log import load_logged_sentiment_urls, append_sentiment_log
from lstm_model import get_lstm_predictions
from lstm_predictions_log import append_lstm_predictions_log
from account_strategy import apply_account_decay_and_rank
from ranked_opportunities_log import append_ranked_opportunities_log
from trader import execute_best_trade

# Strategy run window in U.S. market time
EASTERN_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
BROKER_TZ = ZoneInfo("Europe/Nicosia")

START_HOUR = 9
START_MINUTE = 30
END_HOUR = 16
END_MINUTE = 0

ACTIVE_WINDOW_HOURS = 24

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "temporary_backup.json"
TEMP_STATE_FILE = BASE_DIR / "temporary_backup.tmp.json"


def rebuild_seen_urls(article_store: dict[str, list[dict]]) -> dict[str, set[str]]:
    seen_urls_by_symbol = {symbol: set() for symbol in SYMBOLS}

    for symbol in SYMBOLS:
        for article in article_store.get(symbol, []):
            url = article.get("url")
            if url:
                seen_urls_by_symbol[symbol].add(url)

    return seen_urls_by_symbol


def parse_iso_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    return dt.astimezone(UTC_TZ)


def prune_article_store(
    article_store: dict[str, list[dict]],
    active_window_hours: int = ACTIVE_WINDOW_HOURS,
) -> dict[str, list[dict]]:
    cutoff = datetime.now(UTC_TZ) - timedelta(hours=active_window_hours)

    for symbol in SYMBOLS:
        articles = article_store.get(symbol, [])
        pruned_articles = []

        for article in articles:
            seen_at_utc = article.get("seen_at_utc")
            if not seen_at_utc:
                continue

            try:
                article_seen_time = parse_iso_utc(seen_at_utc)
            except ValueError:
                continue

            if article_seen_time >= cutoff:
                pruned_articles.append(article)

        pruned_articles.sort(
            key=lambda x: parse_iso_utc(x["seen_at_utc"]),
            reverse=True,
        )

        article_store[symbol] = pruned_articles

    return article_store


def save_state(article_store: dict[str, list[dict]]) -> None:
    data = {
        "article_store": article_store,
        "saved_at_utc": datetime.now(UTC_TZ).isoformat(),
    }

    with open(TEMP_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(TEMP_STATE_FILE, STATE_FILE)


def load_state() -> tuple[dict[str, set[str]], dict[str, list[dict]]]:
    default_article_store = {symbol: [] for symbol in SYMBOLS}
    default_seen_urls = {symbol: set() for symbol in SYMBOLS}

    if not STATE_FILE.exists():
        print("No backup file found. Starting fresh.")
        return default_seen_urls, default_article_store

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        article_store = data.get("article_store", {})

        for symbol in SYMBOLS:
            article_store.setdefault(symbol, [])

        article_store = prune_article_store(
            article_store=article_store,
            active_window_hours=ACTIVE_WINDOW_HOURS,
        )
        seen_urls = rebuild_seen_urls(article_store)

        print(
            f"Loaded backup from {STATE_FILE} | "
            f"stored_articles={sum(len(v) for v in article_store.values())}"
        )
        return seen_urls, article_store

    except Exception as e:
        print(f"Failed to load backup file. Starting fresh. Error: {e}")
        return default_seen_urls, default_article_store


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
    open_positions = get_open_positions()

    if not open_positions:
        print("No open positions found.")
        return

    max_age_seconds = MAX_HOLD_MINUTES * 60

    # Estimate MT5/server clock offset using currently open positions.
    # If pos.time is consistently ahead of local epoch time, compensate for it.
    now_ts = time.time()

    # Only use reasonable positive offsets, not stale/broken values
    candidate_offsets = []
    for pos in open_positions:
        diff = pos.time - now_ts
        if 0 <= diff <= 6 * 3600:
            candidate_offsets.append(diff)

    server_offset_seconds = max(candidate_offsets) if candidate_offsets else 0
    now_server_ts = now_ts + server_offset_seconds

    print(
        f"Close check baseline | now_ts={now_ts} | "
        f"server_offset_seconds={server_offset_seconds:.2f} | "
        f"now_server_ts={now_server_ts}"
    )

    for pos in open_positions:
        try:
            age_seconds = now_server_ts - pos.time

            opened_at_local = datetime.fromtimestamp(pos.time)
            now_server_local = datetime.fromtimestamp(now_server_ts)

            if age_seconds >= max_age_seconds:
                print(
                    f"Closing expired position | "
                    f"symbol={pos.symbol}, ticket={pos.ticket}, age_seconds={age_seconds:.2f}"
                )
                ok = close_position(pos)
                print(f"Close result for ticket {pos.ticket}: {ok}")

        except Exception as e:
            print(
                f"Failed to process/close position "
                f"{getattr(pos, 'ticket', 'unknown')}: {e}"
            )

def run_one_cycle(
    seen_urls: dict[str, set[str]],
    article_store: dict[str, list[dict]],
    logged_sentiment_urls: set[str],
) -> None:
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
        active_window_hours=ACTIVE_WINDOW_HOURS,
    )

    save_state(article_store)

    sentiment_scores = compute_sentiment_scores(news_by_symbol)

    appended_count = append_sentiment_log(
        scored_news_by_symbol=sentiment_scores,
        logged_urls=logged_sentiment_urls,
        logged_at_utc=datetime.now(UTC_TZ).isoformat(),
    )
    print(f"Sentiment log appended rows: {appended_count}")

    lstm_predictions = get_lstm_predictions(SYMBOLS)

    prediction_rows = append_lstm_predictions_log(
        lstm_predictions=lstm_predictions,
        cycle_time_utc=datetime.now(UTC_TZ).isoformat(),
    )
    print(f"LSTM predictions log appended rows: {prediction_rows}")

    # --------------------------------------------------
    # 2. Run account-specific strategy logic
    # --------------------------------------------------
    for account in ACCOUNTS:
        try:
            print(f"Starting account: {account['name']}")

            login_to_account(account)
            print(f"Logged into: {account['name']}")

            close_expired_positions_for_account()

            ranked_opportunities = apply_account_decay_and_rank(
                symbols=SYMBOLS,
                news_by_symbol=news_by_symbol,
                sentiment_scores=sentiment_scores,
                lstm_predictions=lstm_predictions,
                account=account,
            )

            ranked_rows = append_ranked_opportunities_log(
                ranked_opportunities=ranked_opportunities,
                cycle_time_utc=datetime.now(UTC_TZ).isoformat(),
            )
            print(f"Ranked opportunities log appended rows: {ranked_rows}")

            execute_best_trade(account, ranked_opportunities)

            print(account["name"], ranked_opportunities[:3])

        except Exception as e:
            print(f"Account failed: {account['name']} | Error: {e}")

    save_state(article_store)


def main():
    initialize_mt5()

    seen_urls, article_store = load_state()
    logged_sentiment_urls = load_logged_sentiment_urls()

    while True:
        try:
            sleep_until_next_run(CHECK_EVERY_MINUTES)
            run_one_cycle(seen_urls, article_store, logged_sentiment_urls)
        except Exception as e:
            print(f"Cycle crashed: {e}")

            try:
                save_state(article_store)
                print("Saved crash backup.")
            except Exception as save_error:
                print(f"Failed to save crash backup: {save_error}")

            time.sleep(5)


if __name__ == "__main__":
    main()
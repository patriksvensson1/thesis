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
from account_strategy import apply_account_decay_and_rank
from ranked_opportunities_log import append_ranked_opportunities_log
from trader import execute_best_trade

# Strategy run window in U.S. market time
EASTERN_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

START_HOUR = 9
START_MINUTE = 30
END_HOUR = 16
END_MINUTE = 0

ACTIVE_WINDOW_HOURS = 24

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "temporary_backup.json"
TEMP_STATE_FILE = BASE_DIR / "temporary_backup.tmp.json"

# Tiny persistent store for trade open timestamps
TRADE_STATE_FILE = BASE_DIR / "open_trade_times.json"
TEMP_TRADE_STATE_FILE = BASE_DIR / "open_trade_times.tmp.json"


def utc_now_str() -> str:
    return datetime.now(UTC_TZ).strftime("%Y-%m-%d %H:%M:%S UTC")


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
        print(f"[{utc_now_str()}] No backup file found. Starting fresh.")
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
            f"[{utc_now_str()}] Loaded backup from {STATE_FILE} | "
            f"stored_articles={sum(len(v) for v in article_store.values())}"
        )
        return seen_urls, article_store

    except Exception as e:
        print(f"[{utc_now_str()}] Failed to load backup file. Starting fresh. Error: {e}")
        return default_seen_urls, default_article_store


def get_account_key(account: dict) -> str:
    if "login" in account:
        return str(account["login"])
    return str(account["name"])


def load_trade_state() -> dict[str, dict[str, float]]:
    if not TRADE_STATE_FILE.exists():
        print(f"[{utc_now_str()}] No trade timer file found. Starting fresh.")
        return {}

    try:
        with open(TRADE_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        normalized: dict[str, dict[str, float]] = {}

        for account_key, ticket_map in data.items():
            if not isinstance(ticket_map, dict):
                continue

            normalized[account_key] = {}
            for ticket, opened_at_epoch in ticket_map.items():
                try:
                    normalized[account_key][str(ticket)] = float(opened_at_epoch)
                except (TypeError, ValueError):
                    continue

        total_trades = sum(len(v) for v in normalized.values())
        print(
            f"[{utc_now_str()}] Loaded trade timer file from {TRADE_STATE_FILE} | "
            f"tracked_open_trades={total_trades}"
        )
        return normalized

    except Exception as e:
        print(f"[{utc_now_str()}] Failed to load trade timer file. Starting fresh. Error: {e}")
        return {}


def save_trade_state(trade_state: dict[str, dict[str, float]]) -> None:
    with open(TEMP_TRADE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(trade_state, f, ensure_ascii=False, indent=2)

    os.replace(TEMP_TRADE_STATE_FILE, TRADE_STATE_FILE)


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
    Schedule the next run on fixed boundaries inside the market window.

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
    Prints the next run in UTC.
    """
    next_run = get_next_run_time(interval_minutes)
    now_et = datetime.now(EASTERN_TZ)
    sleep_seconds = (next_run - now_et).total_seconds()

    if sleep_seconds > 0:
        next_run_utc = next_run.astimezone(UTC_TZ)
        print(
            f"[{utc_now_str()}] Sleeping until next cycle at "
            f"{next_run_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        time.sleep(sleep_seconds)


def close_expired_positions_for_account(
    account: dict,
    trade_state: dict[str, dict[str, float]],
) -> None:
    open_positions = get_open_positions()

    account_key = get_account_key(account)
    account_trade_times = trade_state.setdefault(account_key, {})
    max_age_seconds = MAX_HOLD_MINUTES * 60
    now_ts = time.time()

    if not open_positions:
        if account_trade_times:
            print(
                f"[{utc_now_str()}] No open positions found for {account['name']}. "
                f"Clearing {len(account_trade_times)} stale tracked trades."
            )
            trade_state[account_key] = {}
        else:
            print(f"[{utc_now_str()}] No open positions found for {account['name']}.")
        return

    open_tickets = {str(pos.ticket) for pos in open_positions}

    stale_tickets = [ticket for ticket in account_trade_times if ticket not in open_tickets]
    for ticket in stale_tickets:
        del account_trade_times[ticket]

    for pos in open_positions:
        try:
            ticket = str(pos.ticket)
            opened_at_epoch = account_trade_times.get(ticket)

            if opened_at_epoch is None:
                print(
                    f"[{utc_now_str()}] Skipping untracked position | "
                    f"account={account['name']} | symbol={pos.symbol} | ticket={pos.ticket}"
                )
                continue

            age_seconds = now_ts - opened_at_epoch

            if age_seconds >= max_age_seconds:
                print(
                    f"[{utc_now_str()}] Closing expired position | "
                    f"account={account['name']} | "
                    f"symbol={pos.symbol} | "
                    f"ticket={pos.ticket} | "
                    f"age_seconds={age_seconds:.2f}"
                )
                ok = close_position(pos)
                print(f"[{utc_now_str()}] Close result for ticket {pos.ticket}: {ok}")

                if ok:
                    account_trade_times.pop(ticket, None)

        except Exception as e:
            print(
                f"[{utc_now_str()}] Failed to process/close position "
                f"{getattr(pos, 'ticket', 'unknown')}: {e}"
            )


def run_one_cycle(
    seen_urls: dict[str, set[str]],
    article_store: dict[str, list[dict]],
    logged_sentiment_urls: set[str],
    trade_state: dict[str, dict[str, float]],
) -> None:
    """
    Run one full strategy cycle.
    """
    print(f"\n[{utc_now_str()}] Running cycle")

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
    print(f"[{utc_now_str()}] Sentiment log appended rows: {appended_count}")

    lstm_predictions = get_lstm_predictions(SYMBOLS)

    # --------------------------------------------------
    # 2. Run account-specific strategy logic
    # --------------------------------------------------
    for account in ACCOUNTS:
        try:
            print(f"[{utc_now_str()}] Starting account: {account['name']}")

            login_to_account(account)
            print(f"[{utc_now_str()}] Logged into: {account['name']}")

            close_expired_positions_for_account(account, trade_state)

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
            print(f"[{utc_now_str()}] Ranked opportunities log appended rows: {ranked_rows}")

            execute_best_trade(account, ranked_opportunities, trade_state)

            print(f"[{utc_now_str()}] {account['name']} {ranked_opportunities[:3]}")

        except Exception as e:
            print(f"[{utc_now_str()}] Account failed: {account['name']} | Error: {e}")

    save_state(article_store)
    save_trade_state(trade_state)


def main():
    initialize_mt5()

    seen_urls, article_store = load_state()
    logged_sentiment_urls = load_logged_sentiment_urls()
    trade_state = load_trade_state()

    while True:
        try:
            sleep_until_next_run(CHECK_EVERY_MINUTES)
            run_one_cycle(
                seen_urls=seen_urls,
                article_store=article_store,
                logged_sentiment_urls=logged_sentiment_urls,
                trade_state=trade_state,
            )
        except Exception as e:
            print(f"[{utc_now_str()}] Cycle crashed: {e}")

            try:
                save_state(article_store)
                save_trade_state(trade_state)
                print(f"[{utc_now_str()}] Saved crash backup.")
            except Exception as save_error:
                print(f"[{utc_now_str()}] Failed to save crash backup: {save_error}")

            time.sleep(5)


if __name__ == "__main__":
    main()
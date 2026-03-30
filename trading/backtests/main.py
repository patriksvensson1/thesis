from __future__ import annotations

from pathlib import Path
from datetime import timezone
from typing import Any

import pandas as pd

from config import SYMBOLS, ACCOUNTS, MAX_HOLD_MINUTES
from sentiment import compute_sentiment_scores
from account_strategy import apply_account_decay_and_rank

from backtest_data import (
    load_historical_prices,
    load_historical_news,
    build_backtest_calendar,
    get_news_snapshot,
    build_m15_from_m5,
)
from backtest_lstm import get_lstm_predictions_from_history
from backtest_broker import (
    create_account_states,
    close_expired_positions_for_account,
    close_positions_hit_sl_tp,
    execute_best_trade_backtest,
    mark_account_equity,
)
from backtest_logs import (
    append_lstm_predictions_backtest_log,
    append_ranked_opportunities_backtest_log,
    save_backtest_results,
)

YEAR = 2023

UTC = timezone.utc
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/cleaned_data"
OUTPUT_DIR = BASE_DIR / f"data/backtest_output_{YEAR}"

PRICES_FILE = DATA_DIR / f"cleaned_prices_{YEAR}.csv"
NEWS_FILES = [
    DATA_DIR / f"cleaned_news_{YEAR}_1.csv",
    DATA_DIR / f"cleaned_news_{YEAR}_2.csv",
]

ACTIVE_WINDOW_HOURS = 24


def is_m15_boundary(current_time: pd.Timestamp) -> bool:
    return current_time.minute % 15 == 0


def persist_results(
    output_dir: Path,
    account_states: list[dict[str, Any]],
    lstm_log_rows: list[dict[str, Any]],
    ranked_log_rows: list[dict[str, Any]],
) -> None:
    save_backtest_results(
        output_dir=output_dir,
        account_states=account_states,
        lstm_log_rows=lstm_log_rows,
        ranked_log_rows=ranked_log_rows,
    )


def load_all_historical_news(news_files: list[Path], symbols: list[str]) -> pd.DataFrame:
    news_parts: list[pd.DataFrame] = []

    for news_file in news_files:
        if not news_file.exists():
            print(f"News file not found: {news_file}")
            continue

        part = load_historical_news(news_file, symbols=symbols)
        print(f"Loaded news file: {news_file} | rows={len(part)}")
        news_parts.append(part)

    if not news_parts:
        raise FileNotFoundError("No cleaned news files were found.")

    news_df = pd.concat(news_parts, ignore_index=True)

    if "seen_at_utc" in news_df.columns:
        news_df = news_df.sort_values(
            by=["symbol", "seen_at_utc", "url"],
            na_position="last"
        ).reset_index(drop=True)

    print(f"Total combined news rows: {len(news_df)}")
    return news_df


def run_m5_step(
    current_time: pd.Timestamp,
    prices_m5_df: pd.DataFrame,
    account_states: list[dict[str, Any]],
) -> None:
    for account_state in account_states:
        close_positions_hit_sl_tp(
            account_state=account_state,
            current_time=current_time,
            prices_df=prices_m5_df,
        )

        close_expired_positions_for_account(
            account_state=account_state,
            current_time=current_time,
            prices_df=prices_m5_df,
            max_hold_minutes=MAX_HOLD_MINUTES,
        )

        mark_account_equity(
            account_state=account_state,
            current_time=current_time,
            prices_df=prices_m5_df,
        )


def run_one_cycle(
    current_time: pd.Timestamp,
    prices_m15_df: pd.DataFrame,
    prices_m5_df: pd.DataFrame,
    news_df: pd.DataFrame,
    account_states: list[dict[str, Any]],
    lstm_log_rows: list[dict[str, Any]],
    ranked_log_rows: list[dict[str, Any]],
) -> None:
    print(f"\nRunning M15 signal cycle at {current_time}")

    news_by_symbol = get_news_snapshot(
        news_df=news_df,
        current_time=current_time,
        symbols=SYMBOLS,
        active_window_hours=ACTIVE_WINDOW_HOURS,
    )

    sentiment_scores = compute_sentiment_scores(news_by_symbol)

    lstm_predictions = get_lstm_predictions_from_history(
        symbols=SYMBOLS,
        prices_df=prices_m15_df,
        current_time=current_time,
    )

    append_lstm_predictions_backtest_log(
        lstm_predictions=lstm_predictions,
        cycle_time_utc=current_time,
        rows_accumulator=lstm_log_rows,
    )

    for account_state in account_states:
        account = account_state["config"]

        try:
            ranked_opportunities = apply_account_decay_and_rank(
                symbols=SYMBOLS,
                news_by_symbol=news_by_symbol,
                sentiment_scores=sentiment_scores,
                lstm_predictions=lstm_predictions,
                account=account,
                current_time_utc=current_time,
            )

            append_ranked_opportunities_backtest_log(
                ranked_opportunities=ranked_opportunities,
                cycle_time_utc=current_time,
                rows_accumulator=ranked_log_rows,
            )

            execute_best_trade_backtest(
                account_state=account_state,
                ranked_opportunities=ranked_opportunities,
                current_time=current_time,
                prices_df=prices_m5_df,
            )

            print(account["name"], ranked_opportunities[:3])

        except Exception as e:
            print(f"Account failed: {account['name']} | Error: {e}")


def main() -> None:
    prices_m5_df = load_historical_prices(PRICES_FILE, symbols=SYMBOLS)
    prices_m15_df = build_m15_from_m5(prices_m5_df)
    news_df = load_all_historical_news(NEWS_FILES, symbols=SYMBOLS)

    m5_calendar = build_backtest_calendar(prices_m5_df)
    account_states = create_account_states(ACCOUNTS)

    lstm_log_rows: list[dict[str, Any]] = []
    ranked_log_rows: list[dict[str, Any]] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for current_time in m5_calendar:
        try:
            run_m5_step(
                current_time=current_time,
                prices_m5_df=prices_m5_df,
                account_states=account_states,
            )

            if is_m15_boundary(current_time):
                run_one_cycle(
                    current_time=current_time,
                    prices_m15_df=prices_m15_df,
                    prices_m5_df=prices_m5_df,
                    news_df=news_df,
                    account_states=account_states,
                    lstm_log_rows=lstm_log_rows,
                    ranked_log_rows=ranked_log_rows,
                )

            persist_results(
                output_dir=OUTPUT_DIR,
                account_states=account_states,
                lstm_log_rows=lstm_log_rows,
                ranked_log_rows=ranked_log_rows,
            )

        except Exception as e:
            print(f"Cycle crashed at {current_time}: {e}")

            try:
                persist_results(
                    output_dir=OUTPUT_DIR,
                    account_states=account_states,
                    lstm_log_rows=lstm_log_rows,
                    ranked_log_rows=ranked_log_rows,
                )
            except Exception as save_error:
                print(f"Failed to save partial results at {current_time}: {save_error}")

    persist_results(
        output_dir=OUTPUT_DIR,
        account_states=account_states,
        lstm_log_rows=lstm_log_rows,
        ranked_log_rows=ranked_log_rows,
    )

    print("Backtest complete.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
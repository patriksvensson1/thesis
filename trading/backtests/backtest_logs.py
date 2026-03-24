from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def append_lstm_predictions_backtest_log(
    lstm_predictions: dict[str, dict],
    cycle_time_utc: pd.Timestamp,
    rows_accumulator: list[dict[str, Any]],
) -> int:
    """
    Add one row per symbol for the current backtest cycle.

    Returns the number of rows appended.
    """
    appended = 0

    for symbol, pred in lstm_predictions.items():
        rows_accumulator.append(
            {
                "cycle_time_utc": cycle_time_utc,
                "symbol": symbol,
                "prob_up": pred.get("prob_up"),
                "predicted_class": pred.get("predicted_class"),
            }
        )
        appended += 1

    return appended


def append_ranked_opportunities_backtest_log(
    ranked_opportunities: list[dict[str, Any]],
    cycle_time_utc: pd.Timestamp,
    rows_accumulator: list[dict[str, Any]],
) -> int:
    """
    Add one row per ranked opportunity for the current backtest cycle.

    Returns the number of rows appended.
    """
    appended = 0

    for rank, row in enumerate(ranked_opportunities, start=1):
        rows_accumulator.append(
            {
                "cycle_time_utc": cycle_time_utc,
                "rank": rank,
                **row,
            }
        )
        appended += 1

    return appended


def save_backtest_results(
    output_dir: str | Path,
    account_states: list[dict[str, Any]],
    lstm_log_rows: list[dict[str, Any]],
    ranked_log_rows: list[dict[str, Any]],
) -> None:
    """
    Save all backtest outputs to CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if lstm_log_rows:
        pd.DataFrame(lstm_log_rows).to_csv(
            output_dir / "lstm_predictions_backtest.csv",
            index=False,
        )

    if ranked_log_rows:
        pd.DataFrame(ranked_log_rows).to_csv(
            output_dir / "ranked_opportunities_backtest.csv",
            index=False,
        )

    closed_trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for account_state in account_states:
        account_name = account_state["config"]["name"]

        for pos in account_state["closed_positions"]:
            closed_trade_rows.append(
                {
                    "account_name": account_name,
                    "symbol": pos["symbol"],
                    "side": pos["side"],
                    "entry_time": pos["entry_time"],
                    "entry_price": pos["entry_price"],
                    "exit_time": pos["exit_time"],
                    "exit_price": pos["exit_price"],
                    "exit_reason": pos["exit_reason"],
                    "volume": pos["volume"],
                    "stop_loss": pos["stop_loss"],
                    "take_profit": pos["take_profit"],
                    "score": pos["score"],
                    "pnl": pos["pnl"],
                }
            )

        equity_rows.extend(account_state["equity_curve"])

        closed = account_state["closed_positions"]
        total_pnl = sum((pos.get("pnl") or 0.0) for pos in closed)
        wins = sum(1 for pos in closed if (pos.get("pnl") or 0.0) > 0)
        losses = sum(1 for pos in closed if (pos.get("pnl") or 0.0) < 0)

        final_cash = account_state["cash"]
        final_equity = (
            account_state["equity_curve"][-1]["equity"]
            if account_state["equity_curve"]
            else final_cash
        )

        summary_rows.append(
            {
                "account_name": account_name,
                "final_cash": final_cash,
                "final_equity": final_equity,
                "closed_trades": len(closed),
                "wins": wins,
                "losses": losses,
                "total_pnl": total_pnl,
            }
        )

    if closed_trade_rows:
        pd.DataFrame(closed_trade_rows).to_csv(
            output_dir / "closed_trades_backtest.csv",
            index=False,
        )

    if equity_rows:
        pd.DataFrame(equity_rows).to_csv(
            output_dir / "equity_curve_backtest.csv",
            index=False,
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "summary_backtest.csv",
            index=False,
        )
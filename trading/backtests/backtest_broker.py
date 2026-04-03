from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


INITIAL_CASH = 50000.0

STOP_LOSS_PCT = 0.003
TAKE_PROFIT_PCT = 0.006

MAX_OPEN_TRADES = 8
MAX_TOTAL_RISK_PCT = 0.01


@dataclass
class Position:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    volume: float
    stop_loss: float
    take_profit: float
    score: float

    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl: float | None = None


@dataclass
class AccountState:
    config: dict[str, Any]
    cash: float = INITIAL_CASH
    open_positions: list[Position] = field(default_factory=list)
    closed_positions: list[Position] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)


def create_account_states(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []

    for account in accounts:
        state = AccountState(config=account)
        states.append(
            {
                "config": state.config,
                "cash": state.cash,
                "open_positions": state.open_positions,
                "closed_positions": state.closed_positions,
                "equity_curve": state.equity_curve,
            }
        )

    return states


def _get_bar(
    prices_df: pd.DataFrame,
    symbol: str,
    current_time: pd.Timestamp,
) -> pd.Series | None:
    rows = prices_df[
        (prices_df["symbol"] == symbol) &
        (prices_df["time"] == current_time)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def _get_next_bar(
    prices_df: pd.DataFrame,
    symbol: str,
    current_time: pd.Timestamp,
) -> pd.Series | None:
    rows = prices_df[
        (prices_df["symbol"] == symbol) &
        (prices_df["time"] > current_time)
    ].sort_values("time")

    if rows.empty:
        return None

    return rows.iloc[0]

def _get_last_bar(
    prices_df: pd.DataFrame,
    symbol: str,
) -> pd.Series | None:
    rows = prices_df[
        prices_df["symbol"] == symbol
    ].sort_values("time")

    if rows.empty:
        return None

    return rows.iloc[-1]

def _calculate_sl_tp(entry_price: float, action: str) -> tuple[float, float]:
    if action == "buy":
        stop_loss = entry_price * (1.0 - STOP_LOSS_PCT)
        take_profit = entry_price * (1.0 + TAKE_PROFIT_PCT)
    elif action == "sell":
        stop_loss = entry_price * (1.0 + STOP_LOSS_PCT)
        take_profit = entry_price * (1.0 - TAKE_PROFIT_PCT)
    else:
        raise ValueError(f"Unsupported action: {action}")

    return stop_loss, take_profit


def _calculate_volume_from_risk(
    cash: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    allowed_risk_money = (cash * MAX_TOTAL_RISK_PCT) / MAX_OPEN_TRADES
    price_risk = abs(entry_price - stop_loss)

    if price_risk <= 0:
        return 0.0

    return allowed_risk_money / price_risk


def _compute_pnl(pos: dict[str, Any], exit_price: float) -> float:
    if pos["side"] == "buy":
        return (exit_price - pos["entry_price"]) * pos["volume"]
    if pos["side"] == "sell":
        return (pos["entry_price"] - exit_price) * pos["volume"]
    raise ValueError(f"Unsupported side: {pos['side']}")


def _close_position(
    account_state: dict[str, Any],
    pos: dict[str, Any],
    exit_time: pd.Timestamp,
    exit_price: float,
    reason: str,
) -> None:
    pnl = _compute_pnl(pos, exit_price)

    pos["exit_time"] = exit_time
    pos["exit_price"] = float(exit_price)
    pos["exit_reason"] = reason
    pos["pnl"] = float(pnl)

    account_state["cash"] += pnl
    account_state["closed_positions"].append(pos)


def close_positions_hit_sl_tp(
    account_state: dict[str, Any],
    current_time: pd.Timestamp,
    prices_df: pd.DataFrame,
) -> None:
    """
    Check the current M5 candle for SL/TP hits.

    Assumption:
    - If both SL and TP are inside the same candle, SL is checked first.
    - A position cannot be closed on its entry bar.
    """
    still_open: list[dict[str, Any]] = []

    for pos in account_state["open_positions"]:
        if pos["entry_time"] == current_time:
            still_open.append(pos)
            continue

        bar = _get_bar(prices_df, pos["symbol"], current_time)

        if bar is None:
            still_open.append(pos)
            continue

        low = float(bar["low"])
        high = float(bar["high"])

        closed = False

        if pos["side"] == "buy":
            if low <= pos["stop_loss"]:
                _close_position(account_state, pos, current_time, pos["stop_loss"], "stop_loss")
                closed = True
            elif high >= pos["take_profit"]:
                _close_position(account_state, pos, current_time, pos["take_profit"], "take_profit")
                closed = True

        elif pos["side"] == "sell":
            if high >= pos["stop_loss"]:
                _close_position(account_state, pos, current_time, pos["stop_loss"], "stop_loss")
                closed = True
            elif low <= pos["take_profit"]:
                _close_position(account_state, pos, current_time, pos["take_profit"], "take_profit")
                closed = True

        if not closed:
            still_open.append(pos)

    account_state["open_positions"] = still_open


def close_expired_positions_for_account(
    account_state: dict[str, Any],
    current_time: pd.Timestamp,
    prices_df: pd.DataFrame,
    max_hold_minutes: int,
) -> None:
    """
    Close positions whose holding time has reached the configured limit.
    Exit price is the current bar open.
    A position cannot be closed on its entry bar.
    """
    still_open: list[dict[str, Any]] = []

    for pos in account_state["open_positions"]:
        if pos["entry_time"] == current_time:
            still_open.append(pos)
            continue

        age_minutes = (current_time - pos["entry_time"]).total_seconds() / 60.0

        if age_minutes < max_hold_minutes:
            still_open.append(pos)
            continue

        bar = _get_bar(prices_df, pos["symbol"], current_time)

        if bar is None:
            still_open.append(pos)
            continue

        exit_price = float(bar["open"])
        _close_position(account_state, pos, current_time, exit_price, "max_hold")

    account_state["open_positions"] = still_open

def close_all_open_positions_at_end(
    account_state: dict[str, Any],
    prices_df: pd.DataFrame,
) -> None:
    """
    Force-close any remaining open positions at the last available M5 close
    for each symbol. Exit reason is end_of_year.
    """
    still_open: list[dict[str, Any]] = []

    for pos in account_state["open_positions"]:
        last_bar = _get_last_bar(prices_df, pos["symbol"])

        if last_bar is None:
            still_open.append(pos)
            continue

        exit_time = pd.Timestamp(last_bar["time"])
        exit_price = float(last_bar["close"])

        _close_position(
            account_state=account_state,
            pos=pos,
            exit_time=exit_time,
            exit_price=exit_price,
            reason="end_of_year",
        )

    account_state["open_positions"] = still_open


def execute_best_trade_backtest(
    account_state: dict[str, Any],
    ranked_opportunities: list[dict[str, Any]],
    current_time: pd.Timestamp,
    prices_df: pd.DataFrame,
) -> None:
    """
    Open at most one trade per signal cycle.

    Entry convention:
    signal is generated at current_time,
    order fills at the next M5 bar open.
    """
    if not ranked_opportunities:
        return

    if len(account_state["open_positions"]) >= MAX_OPEN_TRADES:
        return

    best_trade = ranked_opportunities[0]
    action = best_trade.get("action")
    symbol = best_trade.get("symbol")

    if action not in {"buy", "sell"}:
        return

    next_bar = _get_next_bar(prices_df, symbol, current_time)
    if next_bar is None:
        return

    entry_time = pd.Timestamp(next_bar["time"])
    entry_price = float(next_bar["open"])

    stop_loss, take_profit = _calculate_sl_tp(entry_price, action)
    volume = _calculate_volume_from_risk(
        cash=account_state["cash"],
        entry_price=entry_price,
        stop_loss=stop_loss,
    )

    if volume <= 0:
        return

    pos = {
        "symbol": symbol,
        "side": action,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "volume": float(volume),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "score": float(best_trade["final_score"]),
        "exit_time": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl": None,
    }

    account_state["open_positions"].append(pos)


def mark_account_equity(
    account_state: dict[str, Any],
    current_time: pd.Timestamp,
    prices_df: pd.DataFrame,
) -> None:
    unrealized = 0.0

    for pos in account_state["open_positions"]:
        bar = _get_bar(prices_df, pos["symbol"], current_time)
        if bar is None:
            continue

        mark_price = float(bar["close"])
        unrealized += _compute_pnl(pos, mark_price)

    equity = account_state["cash"] + unrealized

    account_state["equity_curve"].append(
        {
            "time": current_time,
            "account_name": account_state["config"]["name"],
            "cash": account_state["cash"],
            "equity": equity,
            "open_positions": len(account_state["open_positions"]),
        }
    )
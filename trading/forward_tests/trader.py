import time
import MetaTrader5 as mt5

STOP_LOSS_PCT = 0.003
TAKE_PROFIT_PCT = 0.005
MAX_OPEN_TRADES = 8
MAX_TOTAL_RISK_PCT = 0.01
DEVIATION = 20
STRATEGY_ID = 20260319


def _build_retcode_name_map() -> dict[int, str]:
    mapping = {}
    for attr_name in dir(mt5):
        if attr_name.startswith("TRADE_RETCODE_"):
            value = getattr(mt5, attr_name)
            if isinstance(value, int):
                mapping[value] = attr_name
    return mapping


RETCODE_NAMES = _build_retcode_name_map()


def _format_retcode(result) -> str:
    if result is None:
        return "None"

    retcode = getattr(result, "retcode", None)
    retcode_name = RETCODE_NAMES.get(retcode, "UNKNOWN_RETCODE")
    comment = getattr(result, "comment", "")

    if comment:
        return f"{retcode} ({retcode_name}) | comment={comment}"
    return f"{retcode} ({retcode_name})"


def get_account_key(account: dict) -> str:
    if "login" in account:
        return str(account["login"])
    return str(account["name"])


def _get_current_price(symbol: str, action: str) -> float | None:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    if action == "buy":
        return float(tick.ask)
    if action == "sell":
        return float(tick.bid)

    return None


def _calculate_sl_tp(entry_price: float, action: str) -> tuple[float, float]:
    if action == "buy":
        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
        take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
    elif action == "sell":
        stop_loss = entry_price * (1 + STOP_LOSS_PCT)
        take_profit = entry_price * (1 - TAKE_PROFIT_PCT)
    else:
        raise ValueError(f"Unsupported action: {action}")

    return stop_loss, take_profit


def _count_open_positions() -> int:
    positions = mt5.positions_get()
    if positions is None:
        return 0
    return len(positions)


def _calculate_risk_per_trade_money() -> float | None:
    account_info = mt5.account_info()
    if account_info is None:
        return None

    balance = float(account_info.balance)
    return (balance * MAX_TOTAL_RISK_PCT) / MAX_OPEN_TRADES


def _calculate_volume_from_risk(
    symbol: str,
    entry_price: float,
    stop_loss: float,
) -> float | None:
    allowed_risk_money = _calculate_risk_per_trade_money()
    if allowed_risk_money is None:
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None

    price_risk = abs(entry_price - stop_loss)
    if price_risk <= 0:
        return None

    contract_size = float(symbol_info.trade_contract_size)
    loss_per_1_lot = price_risk * contract_size

    if loss_per_1_lot <= 0:
        return None

    raw_volume = allowed_risk_money / loss_per_1_lot

    volume_min = float(symbol_info.volume_min)
    volume_max = float(symbol_info.volume_max)
    volume_step = float(symbol_info.volume_step)

    volume = max(volume_min, min(raw_volume, volume_max))

    steps = int(volume / volume_step)
    volume = steps * volume_step

    if volume < volume_min:
        volume = volume_min

    volume = round(volume, 8)

    return volume


def _build_order_request(symbol: str, action: str) -> dict | None:
    entry_price = _get_current_price(symbol, action)
    if entry_price is None:
        return None

    stop_loss, take_profit = _calculate_sl_tp(entry_price, action)
    volume = _calculate_volume_from_risk(symbol, entry_price, stop_loss)

    if volume is None:
        return None

    if action == "buy":
        order_type = mt5.ORDER_TYPE_BUY
    elif action == "sell":
        order_type = mt5.ORDER_TYPE_SELL
    else:
        return None

    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": entry_price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": DEVIATION,
        "magic": STRATEGY_ID,
        "comment": "thesis_trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }


def _get_open_ticket_set() -> set[str]:
    positions = mt5.positions_get()
    if positions is None:
        return set()
    return {str(pos.ticket) for pos in positions}


def _record_new_trade_open_time(
    account: dict,
    trade_state: dict[str, dict[str, float]],
    tickets_before: set[str],
) -> None:
    account_key = get_account_key(account)
    account_trade_times = trade_state.setdefault(account_key, {})

    positions_after = mt5.positions_get()
    if positions_after is None:
        print(f"Could not fetch open positions after trade for {account['name']}")
        return

    tickets_after = {str(pos.ticket) for pos in positions_after}
    new_tickets = tickets_after - tickets_before

    if len(new_tickets) != 1:
        print(
            f"Could not uniquely identify new trade for {account['name']} | "
            f"new_tickets={sorted(new_tickets)}"
        )
        return

    new_ticket = next(iter(new_tickets))
    account_trade_times[new_ticket] = time.time()

    print(
        f"Recorded trade open time | "
        f"account={account['name']} | ticket={new_ticket}"
    )


def execute_best_trade(
    account: dict,
    ranked_opportunities: list[dict],
    trade_state: dict[str, dict[str, float]],
) -> None:
    if not ranked_opportunities:
        print(f"No opportunities for {account['name']}")
        return

    current_open = _count_open_positions()
    if current_open >= MAX_OPEN_TRADES:
        print(
            f"No trade executed for {account['name']} "
            f"(open positions {current_open}/{MAX_OPEN_TRADES})"
        )
        return

    best_trade = ranked_opportunities[0]
    action = best_trade.get("action")
    symbol = best_trade.get("symbol")

    if action not in {"buy", "sell"}:
        print(f"No trade executed for {account['name']} (best action was hold)")
        return

    request = _build_order_request(symbol=symbol, action=action)

    if request is None:
        print(f"Could not build order request for {symbol}")
        return

    tickets_before = _get_open_ticket_set()
    result = mt5.order_send(request)

    if result is None:
        print(f"MT5 order_send returned None for {symbol}")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(
            f"Trade failed for {account['name']} | "
            f"symbol={symbol} | action={action} | {_format_retcode(result)}"
        )
        return

    _record_new_trade_open_time(
        account=account,
        trade_state=trade_state,
        tickets_before=tickets_before,
    )

    print(
        f"Trade executed for {account['name']} | "
        f"symbol={symbol} | action={action} | "
        f"score={best_trade.get('final_score'):.4f} | "
        f"volume={request['volume']} | "
        f"sl={request['sl']:.4f} | tp={request['tp']:.4f}"
    )
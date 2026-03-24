import MetaTrader5 as mt5

STOP_LOSS_PCT = 0.003   # 0.3%
TAKE_PROFIT_PCT = 0.005 # 0.5%

# 2h holding time and 15-minute cycle -> at most 8 overlapping trades
MAX_OPEN_TRADES = 8

# Total account risk cap = 1%
MAX_TOTAL_RISK_PCT = 0.01

DEVIATION = 20
STRATEGY_ID = 20260319


def _get_current_price(symbol: str, action: str) -> float | None:
    """
    Get the current market price for a symbol.

    Buy  -> ask
    Sell -> bid
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    if action == "buy":
        return float(tick.ask)
    if action == "sell":
        return float(tick.bid)

    return None


def _calculate_sl_tp(entry_price: float, action: str) -> tuple[float, float]:
    """
    Calculate stop loss and take profit from entry price.
    """
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
    """
    Count currently open positions in the currently logged-in MT5 account.
    """
    positions = mt5.positions_get()
    if positions is None:
        return 0
    return len(positions)


def _calculate_risk_per_trade_money() -> float | None:
    """
    Total max risk is 1% of account balance.
    If max open trades is 8, each trade may risk at most 1% / 8.
    """
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
    """
    Compute order volume so that the worst-case loss at stop loss is equal to
    the allowed risk per trade.

    This uses MT5 symbol metadata:
    - trade_contract_size
    - volume_min
    - volume_max
    - volume_step

    For many CFDs/stocks, a rough loss estimate is:
        loss_per_1_lot = |entry - stop_loss| * contract_size

    Then:
        volume = allowed_risk_money / loss_per_1_lot
    """
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
    """
    Build the MT5 order request dictionary for a buy or sell order.
    Volume is computed dynamically from the account risk rule.
    """
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

    request = {
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

    return request


def execute_best_trade(account: dict, ranked_opportunities: list[dict]) -> None:
    """
    Execute at most one trade per cycle for the currently logged-in account.

    This matches the thesis logic:
    - one new trade per 15-minute cycle
    - 2-hour max holding time
    - therefore max overlapping trades is naturally 8
    - risk per trade = 1% / 8 of account balance

    Multiple positions in the same symbol are allowed.
    """
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

    request = _build_order_request(
        symbol=symbol,
        action=action,
    )

    if request is None:
        print(f"Could not build order request for {symbol}")
        return

    result = mt5.order_send(request)

    if result is None:
        print(f"MT5 order_send returned None for {symbol}")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(
            f"Trade failed for {account['name']} | "
            f"symbol={symbol} | action={action} | retcode={result.retcode}"
        )
        return

    print(
        f"Trade executed for {account['name']} | "
        f"symbol={symbol} | action={action} | "
        f"score={best_trade.get('final_score'):.4f} | "
        f"volume={request['volume']} | "
        f"sl={request['sl']:.4f} | tp={request['tp']:.4f}"
    )
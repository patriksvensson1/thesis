from __future__ import annotations

import MetaTrader5 as mt5
from config import ACCOUNTS

DEVIATION = 20
MAGIC = 20260319


def initialize_and_login(account: dict) -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    ok = mt5.login(
        login=int(account["login"]),
        password=account["password"],
        server=account["server"],
    )
    if not ok:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"MT5 login failed for {account['name']}: {err}")


def shutdown_mt5() -> None:
    mt5.shutdown()


def close_position(position) -> bool:
    symbol = position.symbol
    volume = float(position.volume)
    ticket = int(position.ticket)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Could not get tick for {symbol}")
        return False

    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    elif position.type == mt5.POSITION_TYPE_SELL:
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)
    else:
        print(f"Unsupported position type for ticket {ticket}")
        return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "final_close_all",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        print(f"order_send returned None for ticket {ticket}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(
            f"Failed to close ticket={ticket} symbol={symbol} "
            f"retcode={result.retcode} comment={getattr(result, 'comment', '')}"
        )
        return False

    print(f"Closed ticket={ticket} symbol={symbol} volume={volume}")
    return True


def close_all_positions_for_current_account() -> None:
    positions = mt5.positions_get()

    if positions is None:
        raise RuntimeError(f"positions_get failed: {mt5.last_error()}")

    if len(positions) == 0:
        print("No open positions.")
        return

    for position in positions:
        close_position(position)


def main() -> None:
    for account in ACCOUNTS:
        print(f"\nLogging into {account['name']}...")
        initialize_and_login(account)
        try:
            close_all_positions_for_current_account()
        finally:
            shutdown_mt5()


if __name__ == "__main__":
    main()
import MetaTrader5 as mt5


def initialize_mt5() -> None:
    """
    Initialize the MT5 terminal connection once.

    Raises:
        RuntimeError: if MT5 cannot be initialized.
    """
    if mt5.terminal_info() is not None:
        return

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def login_to_account(account: dict) -> None:
    """
    Log in to one MT5 account.

    Expected account format:
        {
            "login": 12345678,
            "password": "your_password",
            "server": "PepperstoneUK-Demo"
        }

    If already logged into this exact account, do nothing.
    """

    initialize_mt5()

    login = int(account["login"])
    password = account["password"]
    server = account["server"]

    current_info = mt5.account_info()

    # Already on the correct account -> no need to switch
    if current_info is not None and int(current_info.login) == login:
        return

    ok = mt5.login(
        login=login,
        password=password,
        server=server,
    )

    if not ok:
        raise RuntimeError(
            f"MT5 login failed for account {login} on server {server}: {mt5.last_error()}"
        )

    # Extra safety check after login
    current_info = mt5.account_info()
    if current_info is None or int(current_info.login) != login:
        raise RuntimeError(
            f"MT5 login appeared to succeed, but active account is not {login}"
        )
    

def get_open_positions(symbol: str | None = None):
    """
    Return open positions for the currently logged-in account.

    If symbol is given, only return positions for that symbol.
    Returns an empty list if nothing is open.
    """
    if symbol is not None:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return []

    return list(positions)

def close_position(pos) -> bool:
    """
    Close one specific open MT5 position.

    Expects an MT5 position object with fields like:
        pos.symbol
        pos.ticket
        pos.volume
        pos.type

    Returns:
        True if close was successful, False otherwise.
    """
    symbol = pos.symbol
    ticket = pos.ticket
    volume = float(pos.volume)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Could not get tick data for {symbol}")
        return False

    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    elif pos.type == mt5.POSITION_TYPE_SELL:
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
        "deviation": 20,
        "comment": "time_limit_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"MT5 order_send returned None while closing ticket {ticket}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(
            f"Failed to close ticket {ticket} | "
            f"symbol={symbol} | retcode={result.retcode}"
        )
        return False

    print(f"Closed ticket {ticket} on {symbol}")
    return True

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

LOGIN = "demo_user_number"
PASSWORD = "demo_user_password"
SERVER = "demo_user_server"

symbols = ["AAPL.US", "MSFT.US", "NVDA.US", "AMZN.US", "GOOGL.US",
           "NFLX.US", "WBD.US", "TSLA.US", "NDAQ.US", "SBUX.US",
           "ADBE.US", "META.US", "NKE.US", "CRM.US", "PYPL.US"]
timeframe = mt5.TIMEFRAME_M15

date_from = datetime(2024, 3, 1, tzinfo=timezone.utc)
date_to   = datetime(2026, 3, 1, tzinfo=timezone.utc)

all_data = []

if not mt5.initialize():
    print("initialize failed:", mt5.last_error())
    raise SystemExit

if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
    print("login failed:", mt5.last_error())
    mt5.shutdown()
    raise SystemExit

for symbol in symbols:
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}: {mt5.last_error()}")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol
    all_data.append(df)

mt5.shutdown()

prices = pd.concat(all_data, ignore_index=True)
prices.to_csv("raw_price_data_M15.csv", index=False)
print("raw_price_data_M15.csv")
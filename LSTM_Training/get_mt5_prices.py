import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

LOGIN = "demo_user_number"
PASSWORD = "demo_user_password"
SERVER = "demo_user_server"

symbols = [
    "AAPL.NAS",
    "MSFT.NAS",
    "NVDA.NAS",
    "AMZN.NAS",
    "GOOG.NAS",
    "NFLX.NAS",
    "AMD.NAS",
    "TSLA.NAS",
    "NDAQ.NAS",
    "SBUX.NAS",
    "ADBE.NAS",
    "MVRS.NAS",
    "NKE.NYSE",
    "CRM.NYSE",
    "PYPL.NAS",
]

timeframe = mt5.TIMEFRAME_M15

date_from = datetime(2024, 3, 1, tzinfo=timezone.utc)
date_to = datetime(2026, 3, 1, tzinfo=timezone.utc)

all_data = []

script_dir = Path(__file__).resolve().parent
output_file = script_dir / "raw_price_data_M15.csv"

if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
    print("initialize/login failed:", mt5.last_error())
    raise SystemExit

for symbol in symbols:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Symbol not found: {symbol} | {mt5.last_error()}")
        continue

    if not mt5.symbol_select(symbol, True):
        print(f"Could not select {symbol}: {mt5.last_error()}")
        continue

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}: {mt5.last_error()}")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol

    df = df[[
        "symbol",
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]].copy()

    all_data.append(df)
    print(f"{symbol}: {len(df)} rows | first={df['time'].iloc[0]} | last={df['time'].iloc[-1]}")

mt5.shutdown()

if not all_data:
    print("No data collected.")
    raise SystemExit

prices = pd.concat(all_data, ignore_index=True)
prices = prices.drop_duplicates(subset=["symbol", "time"]).sort_values(["symbol", "time"])
prices.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
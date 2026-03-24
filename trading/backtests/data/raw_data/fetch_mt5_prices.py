import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

LOGIN = "demo_user_number"
PASSWORD = "demo_user_password"
SERVER = "demo_user_server"

SYMBOLS = [
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

TIMEFRAME = mt5.TIMEFRAME_M5
YEARS = [2022, 2023, 2024]

SLEEP_BETWEEN_CALLS = 0.2

BASE_DIR = Path(__file__).resolve().parent


def ensure_output_file(year: int) -> Path:
    output_file = BASE_DIR / f"raw_price_data_{year}.csv"

    pd.DataFrame(columns=[
        "symbol", "time", "open", "high", "low", "close",
        "tick_volume", "spread", "real_volume"
    ]).to_csv(output_file, index=False)

    return output_file


def append_prices(output_file: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    df.to_csv(output_file, mode="a", header=False, index=False)


def fetch_symbol_day(symbol: str, day_start: datetime, day_end: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, TIMEFRAME, day_start, day_end)

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol

    df = df[[
        "symbol", "time", "open", "high", "low", "close",
        "tick_volume", "spread", "real_volume"
    ]].copy()

    df = df[(df["time"] >= day_start) & (df["time"] < day_end)].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["symbol", "time"]).sort_values(["symbol", "time"])
    return df


def fetch_one_year(symbols: list[str], year: int) -> None:
    output_file = ensure_output_file(year)

    print(f"\nFetching raw M5 data day-by-day for {year}...")
    start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    for symbol in symbols:
        print(f"\nStarting {symbol} for {year}...")

        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"Could not find {symbol}: {mt5.last_error()}")
            continue

        if not mt5.symbol_select(symbol, True):
            print(f"Could not select {symbol}: {mt5.last_error()}")
            continue

        current_day = start_date

        while current_day < end_date:
            next_day = current_day + timedelta(days=1)
            day_str = current_day.strftime("%Y-%m-%d")

            print(f"Fetching {symbol} {day_str} ...")

            df = fetch_symbol_day(symbol, current_day, next_day)

            if df.empty:
                print(f"  No rows for {symbol} {day_str}")
            else:
                print(
                    f"  Rows={len(df)} | first={df['time'].iloc[0]} | last={df['time'].iloc[-1]}"
                )
                append_prices(output_file, df)

            time.sleep(SLEEP_BETWEEN_CALLS)
            current_day = next_day

    print(f"\nFinished {year}. Prices: {output_file}")


def main():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        print("initialize/login failed:", mt5.last_error())
        raise SystemExit

    account_info = mt5.account_info()
    if account_info is None:
        print("Could not read account info:", mt5.last_error())
        mt5.shutdown()
        raise SystemExit

    print("Connected")
    print("Login:", account_info.login)
    print("Server:", account_info.server)

    try:
        for year in YEARS:
            fetch_one_year(SYMBOLS, year)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
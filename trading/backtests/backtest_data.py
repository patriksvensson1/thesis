from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "symbol",
    "time",
    "open",
    "high",
    "low",
    "close",
}

OPTIONAL_PRICE_COLUMNS_WITH_DEFAULTS = {
    "tick_volume": 0.0,
    "spread": 0.0,
    "real_volume": 0.0,
}

REQUIRED_NEWS_COLUMNS = {
    "symbol",
    "title",
    "url",
    "seen_at_utc",
}

OPTIONAL_NEWS_COLUMNS_WITH_DEFAULTS = {
    "source": None,
    "seendate": None,
    "language": None,
    "sourcecountry": None,
}


def load_historical_prices(path: str | Path, symbols: list[str]) -> pd.DataFrame:
    """
    Load historical M5 prices from CSV.

    Expected columns at minimum:
        symbol,time,open,high,low,close

    Optional columns:
        tick_volume,spread,real_volume
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Historical prices file is missing columns: {sorted(missing)}")

    for col, default_value in OPTIONAL_PRICE_COLUMNS_WITH_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")

    numeric_cols = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["symbol"].isin(symbols)].copy()
    df = df.dropna(subset=["symbol", "time", "open", "high", "low", "close"])
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No historical price rows remain after filtering by symbols.")

    return df


def load_historical_news(path: str | Path, symbols: list[str]) -> pd.DataFrame:
    """
    Load historical news from CSV.

    Expected columns at minimum:
        symbol,title,url,seen_at_utc
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_NEWS_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Historical news file is missing columns: {sorted(missing)}")

    for col, default_value in OPTIONAL_NEWS_COLUMNS_WITH_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["seen_at_utc"] = pd.to_datetime(df["seen_at_utc"], utc=True, errors="raise")

    df = df[df["symbol"].isin(symbols)].copy()
    df = df.dropna(subset=["symbol", "title", "url", "seen_at_utc"])
    df = df.sort_values(["symbol", "seen_at_utc"]).reset_index(drop=True)

    return df


def build_backtest_calendar(prices_df: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Build the master M5 calendar from the timestamps that exist in the price file.
    """
    if "time" not in prices_df.columns:
        raise ValueError("prices_df must contain a 'time' column.")

    calendar = sorted(pd.Series(prices_df["time"]).drop_duplicates().tolist())

    if not calendar:
        raise ValueError("Backtest calendar is empty.")

    return calendar


def build_m15_from_m5(prices_m5_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample M5 OHLCV data into M15 OHLCV data per symbol.

    The resulting timestamps are bar-close timestamps aligned on 15-minute boundaries.
    Example:
        14:45 candle contains the three M5 bars ending by 14:45.
    """
    required = REQUIRED_PRICE_COLUMNS | {"tick_volume", "spread", "real_volume"}
    missing = required - set(prices_m5_df.columns)
    if missing:
        raise ValueError(f"prices_m5_df is missing columns needed for M15 build: {sorted(missing)}")

    out_frames: list[pd.DataFrame] = []

    for symbol, group in prices_m5_df.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        g = g.set_index("time")

        # Resample to 15-minute bars.
        # label='right', closed='right' means a bar timestamp like 14:45
        # represents the bar that just closed at 14:45.
        m15 = g.resample("15min", label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
                "spread": "mean",
                "real_volume": "sum",
            }
        )

        m15 = m15.dropna(subset=["open", "high", "low", "close"]).reset_index()
        m15["symbol"] = symbol

        # Reorder columns to match the style of the source file.
        m15 = m15[
            [
                "symbol",
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            ]
        ]

        out_frames.append(m15)

    if not out_frames:
        raise ValueError("No M15 data could be built from the supplied M5 dataframe.")

    prices_m15_df = pd.concat(out_frames, ignore_index=True)
    prices_m15_df = prices_m15_df.sort_values(["symbol", "time"]).reset_index(drop=True)

    return prices_m15_df


def get_news_snapshot(
    news_df: pd.DataFrame,
    current_time: pd.Timestamp,
    symbols: list[str],
    active_window_hours: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return the news visible up to current_time, limited to the rolling active window.

    Output shape:
        {
            "AAPL.NAS": [article_dict, article_dict, ...],
            "MSFT.NAS": [...],
            ...
        }
    """
    if current_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware.")

    cutoff = current_time - pd.Timedelta(hours=active_window_hours)

    visible = news_df[
        (news_df["symbol"].isin(symbols)) &
        (news_df["seen_at_utc"] <= current_time) &
        (news_df["seen_at_utc"] >= cutoff)
    ].copy()

    result: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}

    if visible.empty:
        return result

    visible = visible.sort_values(["symbol", "seen_at_utc", "url"]).reset_index(drop=True)

    for symbol, group in visible.groupby("symbol", sort=False):
        result[symbol] = group.to_dict("records")

    return result
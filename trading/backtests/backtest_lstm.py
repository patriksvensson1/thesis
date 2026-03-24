from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

MODEL_DIR = PROJECT_ROOT / "trained_lstm"
MODEL_PATH = MODEL_DIR / "lstm_direction_model.keras"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")


_model = tf.keras.models.load_model(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    _metadata = json.load(f)

SEQUENCE_LENGTH = _metadata["sequence_length"]
FEATURE_COLUMNS = _metadata["feature_columns"]


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the same features used during training.
    This MUST match the training feature engineering exactly.
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)

    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["hl_range"] = (df["high"] - df["low"]) / df["low"]
    df["volume_change_1"] = df["tick_volume"].pct_change(1)
    df["volatility_6"] = df["close"].pct_change().rolling(6).std()

    return df


def _get_recent_rates_from_history(
    symbol: str,
    prices_df: pd.DataFrame,
    current_time: pd.Timestamp,
    n_bars: int = 80,
) -> pd.DataFrame:
    """
    Get recent CLOSED historical bars for one symbol up to current_time.

    prices_df is expected to already be M15 data.
    current_time is the simulated timestamp in the backtest.
    """
    if current_time.tzinfo is None:
        raise ValueError("current_time must be timezone-aware.")

    required_cols = {
        "symbol",
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    }
    missing = required_cols - set(prices_df.columns)
    if missing:
        raise ValueError(f"prices_df missing required columns: {sorted(missing)}")

    df = prices_df[
        (prices_df["symbol"] == symbol) &
        (prices_df["time"] <= current_time)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("time").tail(n_bars).reset_index(drop=True)
    return df


def _prepare_latest_sequence_from_history(
    symbol: str,
    prices_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> np.ndarray | None:
    """
    Build the latest sequence for one symbol from historical M15 data.

    Returns:
        ndarray of shape (1, sequence_length, n_features)
        or None if there is not enough valid data.
    """
    df = _get_recent_rates_from_history(
        symbol=symbol,
        prices_df=prices_df,
        current_time=current_time,
        n_bars=max(80, SEQUENCE_LENGTH + 20),
    )

    if df.empty:
        return None

    df = _compute_features(df)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing feature columns {missing}")

    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH:
        return None

    feature_matrix = df[FEATURE_COLUMNS].copy()
    feature_matrix[FEATURE_COLUMNS] = _scaler.transform(feature_matrix[FEATURE_COLUMNS])

    latest_sequence = feature_matrix.iloc[-SEQUENCE_LENGTH:].values.astype(np.float32)
    return np.expand_dims(latest_sequence, axis=0)


def get_lstm_predictions_from_history(
    symbols: list[str],
    prices_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> dict[str, dict]:
    """
    Predict direction for each symbol using historical M15 bars up to current_time.

    Returns:
        {
            "AAPL.NAS": {"prob_up": 0.73, "predicted_class": 1},
            "MSFT.NAS": {"prob_up": 0.41, "predicted_class": 0},
        }
    """
    predictions: dict[str, dict] = {}

    for symbol in symbols:
        try:
            X = _prepare_latest_sequence_from_history(
                symbol=symbol,
                prices_df=prices_df,
                current_time=current_time,
            )

            if X is None:
                predictions[symbol] = {
                    "prob_up": None,
                    "predicted_class": None,
                }
                continue

            prob_up = float(_model.predict(X, verbose=0)[0][0])
            predicted_class = int(prob_up >= 0.5)

            predictions[symbol] = {
                "prob_up": prob_up,
                "predicted_class": predicted_class,
            }

        except Exception as e:
            print(f"LSTM prediction failed for {symbol} at {current_time}: {e}")
            predictions[symbol] = {
                "prob_up": None,
                "predicted_class": None,
            }

    return predictions
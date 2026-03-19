import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import MetaTrader5 as mt5


MODEL_DIR = Path("trained_lstm")
MODEL_PATH = MODEL_DIR / "lstm_direction_model.keras"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")


# Load once when the module is imported
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


def _get_recent_rates(
    symbol: str,
    timeframe: int = mt5.TIMEFRAME_M15,
    n_bars: int = 20,
) -> pd.DataFrame:
    """
    Fetch recent MT5 bars for one symbol.
    """
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol
    return df


def _prepare_latest_sequence(symbol: str) -> np.ndarray | None:
    """
    Build the latest sequence for one symbol.

    Returns:
        ndarray of shape (1, sequence_length, n_features)
        or None if there is not enough valid data.
    """
    df = _get_recent_rates(symbol)

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


def get_lstm_predictions(symbols: list[str]) -> dict[str, dict]:
    """
    Predict direction for each symbol using the trained LSTM.

    Returns:
        {
            "AAPL.US": {"prob_up": 0.73, "predicted_class": 1},
            "MSFT.US": {"prob_up": 0.41, "predicted_class": 0},
        }
    """
    predictions = {}

    for symbol in symbols:
        try:
            X = _prepare_latest_sequence(symbol)

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
            print(f"LSTM prediction failed for {symbol}: {e}")
            predictions[symbol] = {
                "prob_up": None,
                "predicted_class": None,
            }

    return predictions
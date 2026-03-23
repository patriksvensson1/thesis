import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import MetaTrader5 as mt5


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_PRICES_FILE = DATA_DIR / "live_raw_prices_M15.csv"

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


# Load once when the module is imported
_model = tf.keras.models.load_model(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    _metadata = json.load(f)

SEQUENCE_LENGTH = _metadata["sequence_length"]
FEATURE_COLUMNS = _metadata["feature_columns"]


def _ensure_data_dir_and_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PRICES_FILE.exists():
        pd.DataFrame(columns=[
            "symbol",
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
            "fetched_at_utc",
        ]).to_csv(RAW_PRICES_FILE, index=False)


def _append_raw_prices(df: pd.DataFrame) -> None:
    """
    Append raw bars to the CSV in /data, avoiding duplicate symbol+time rows.
    """
    if df.empty:
        return

    _ensure_data_dir_and_file()

    save_df = df.copy()
    save_df["time"] = pd.to_datetime(save_df["time"], utc=True)
    save_df["fetched_at_utc"] = pd.Timestamp.utcnow()

    save_df = save_df[
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
            "fetched_at_utc",
        ]
    ].copy()

    try:
        existing = pd.read_csv(RAW_PRICES_FILE, usecols=["symbol", "time"])
        if not existing.empty:
            existing["time"] = pd.to_datetime(existing["time"], utc=True)

            existing_keys = set(
                zip(existing["symbol"].astype(str), existing["time"].astype(str))
            )

            save_df = save_df[
                ~save_df.apply(
                    lambda row: (str(row["symbol"]), str(row["time"])) in existing_keys,
                    axis=1
                )
            ].copy()
    except Exception:
        pass

    if save_df.empty:
        return

    save_df.to_csv(RAW_PRICES_FILE, mode="a", header=False, index=False)


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
    n_bars: int = 40,
) -> pd.DataFrame:
    """
    Fetch recent CLOSED MT5 bars for one symbol.
    start_pos=1 skips the currently forming bar.
    Also save fetched raw bars into /data/live_raw_prices_M15.csv.
    """
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, n_bars)

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["symbol"] = symbol

    df = df[
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
    ].copy()

    _append_raw_prices(df)

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
            "AAPL.NAS": {"prob_up": 0.73, "predicted_class": 1},
            "MSFT.NAS": {"prob_up": 0.41, "predicted_class": 0},
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
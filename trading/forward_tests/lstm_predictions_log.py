from __future__ import annotations

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


UTC_TZ = ZoneInfo("UTC")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LSTM_PREDICTIONS_LOG_FILE = DATA_DIR / "lstm_predictions_log.csv"


def ensure_lstm_predictions_log_file(
    log_file: Path = LSTM_PREDICTIONS_LOG_FILE,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        pd.DataFrame(columns=[
            "cycle_time_utc",
            "symbol",
            "prob_up",
            "predicted_class",
        ]).to_csv(log_file, index=False)


def append_lstm_predictions_log(
    lstm_predictions: dict[str, dict],
    cycle_time_utc: str | None = None,
    log_file: Path = LSTM_PREDICTIONS_LOG_FILE,
) -> int:
    ensure_lstm_predictions_log_file(log_file)

    if cycle_time_utc is None:
        cycle_time_utc = datetime.now(UTC_TZ).isoformat()

    existing_keys: set[tuple[str, str]] = set()

    try:
        existing = pd.read_csv(log_file, usecols=["cycle_time_utc", "symbol"])
        if not existing.empty:
            existing_keys = set(
                zip(
                    existing["cycle_time_utc"].astype(str),
                    existing["symbol"].astype(str),
                )
            )
    except Exception:
        pass

    rows_to_append: list[dict] = []

    for symbol, prediction in lstm_predictions.items():
        key = (str(cycle_time_utc), str(symbol))

        if key in existing_keys:
            continue

        rows_to_append.append({
            "cycle_time_utc": cycle_time_utc,
            "symbol": symbol,
            "prob_up": prediction.get("prob_up"),
            "predicted_class": prediction.get("predicted_class"),
        })

    if not rows_to_append:
        return 0

    pd.DataFrame(rows_to_append).to_csv(
        log_file,
        mode="a",
        header=False,
        index=False,
    )
    return len(rows_to_append)
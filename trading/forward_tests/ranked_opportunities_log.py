from __future__ import annotations

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


UTC_TZ = ZoneInfo("UTC")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RANKED_OPPORTUNITIES_LOG_FILE = DATA_DIR / "ranked_opportunities_log.csv"


def ensure_ranked_opportunities_log_file(
    log_file: Path = RANKED_OPPORTUNITIES_LOG_FILE,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        pd.DataFrame(columns=[
            "cycle_time_utc",
            "account_name",
            "rank",
            "symbol",
            "news_score",
            "lstm_score",
            "final_score",
            "action",
            "prob_up",
            "predicted_class",
            "article_count",
        ]).to_csv(log_file, index=False)


def append_ranked_opportunities_log(
    ranked_opportunities: list[dict],
    cycle_time_utc: str | None = None,
    log_file: Path = RANKED_OPPORTUNITIES_LOG_FILE,
) -> int:
    ensure_ranked_opportunities_log_file(log_file)

    if cycle_time_utc is None:
        cycle_time_utc = datetime.now(UTC_TZ).isoformat()

    rows_to_append: list[dict] = []

    for rank, item in enumerate(ranked_opportunities, start=1):
        rows_to_append.append({
            "cycle_time_utc": cycle_time_utc,
            "account_name": item.get("account_name"),
            "rank": rank,
            "symbol": item.get("symbol"),
            "news_score": item.get("news_score"),
            "lstm_score": item.get("lstm_score"),
            "final_score": item.get("final_score"),
            "action": item.get("action"),
            "prob_up": item.get("prob_up"),
            "predicted_class": item.get("predicted_class"),
            "article_count": item.get("article_count"),
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
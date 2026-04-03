from pathlib import Path
import pandas as pd
import numpy as np

YEARS = [2022, 2023, 2024]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PRICES_DIR = DATA_DIR / "input_cleaned_data"

AUDIT_OUTPUT_DIR = SCRIPT_DIR
FAILED_AUDIT_FILE = AUDIT_OUTPUT_DIR / "trade_audit_2022_2024_failed.csv"
SUMMARY_AUDIT_FAILED_FILE = AUDIT_OUTPUT_DIR / "summary_audit_2022_2024_failed.csv"

STOP_LOSS_PCT = 0.003
TAKE_PROFIT_PCT = 0.006
MAX_HOLD_MINUTES = 120

PRICE_ATOL = 1e-8
PNL_ATOL = 1e-6


def is_close(a: float, b: float, atol: float = PRICE_ATOL) -> bool:
    return np.isclose(float(a), float(b), atol=atol, rtol=0.0)


def expected_stop_loss(entry_price: float, side: str) -> float:
    if side == "buy":
        return entry_price * (1.0 - STOP_LOSS_PCT)
    if side == "sell":
        return entry_price * (1.0 + STOP_LOSS_PCT)
    raise ValueError(f"Unsupported side: {side}")


def expected_take_profit(entry_price: float, side: str) -> float:
    if side == "buy":
        return entry_price * (1.0 + TAKE_PROFIT_PCT)
    if side == "sell":
        return entry_price * (1.0 - TAKE_PROFIT_PCT)
    raise ValueError(f"Unsupported side: {side}")


def expected_pnl(entry_price: float, exit_price: float, volume: float, side: str) -> float:
    if side == "buy":
        return (exit_price - entry_price) * volume
    if side == "sell":
        return (entry_price - exit_price) * volume
    raise ValueError(f"Unsupported side: {side}")


def load_trades_for_years(years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for year in years:
        trades_file = DATA_DIR / f"output_backtest_{year}" / "closed_trades_backtest.csv"
        if not trades_file.exists():
            raise FileNotFoundError(f"Missing trades file: {trades_file}")

        df = pd.read_csv(trades_file)
        if df.empty:
            continue

        df["source_year"] = year
        frames.append(df)

    if not frames:
        raise ValueError("No trade data loaded from the requested years.")

    return pd.concat(frames, ignore_index=True)


def load_prices_for_years(years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for year in years:
        path = PRICES_DIR / f"cleaned_prices_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing price file: {path}")

        df = pd.read_csv(path)
        if df.empty:
            continue

        df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
        df["symbol"] = df["symbol"].astype(str)
        df["source_year"] = year
        frames.append(df)

    if not frames:
        raise ValueError("No price data loaded from the requested years.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["symbol", "time"]).drop_duplicates(
        subset=["symbol", "time"],
        keep="first"
    ).reset_index(drop=True)
    return prices


def get_bar(price_lookup: pd.DataFrame, symbol: str, ts: pd.Timestamp) -> pd.Series | None:
    try:
        row = price_lookup.loc[(symbol, ts)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    except KeyError:
        return None


def audit_trade(row: pd.Series, price_lookup: pd.DataFrame) -> dict:
    symbol = str(row["symbol"])
    side = str(row["side"]).lower()
    entry_time = row["entry_time"]
    exit_time = row["exit_time"]
    entry_price = float(row["entry_price"])
    exit_price = float(row["exit_price"])
    stop_loss = float(row["stop_loss"])
    take_profit = float(row["take_profit"])
    volume = float(row["volume"])
    pnl = float(row["pnl"])
    exit_reason = str(row["exit_reason"])

    entry_bar = get_bar(price_lookup, symbol, entry_time)
    exit_bar = get_bar(price_lookup, symbol, exit_time)

    checks: dict[str, bool] = {}
    details: list[str] = []

    checks["entry_bar_exists"] = entry_bar is not None
    if not checks["entry_bar_exists"]:
        details.append("missing_entry_bar")

    checks["exit_bar_exists"] = exit_bar is not None
    if not checks["exit_bar_exists"]:
        details.append("missing_exit_bar")

    if exit_reason == "end_of_year":
        checks["exit_after_entry"] = exit_time >= entry_time
    else:
        checks["exit_after_entry"] = exit_time > entry_time

    if not checks["exit_after_entry"]:
        details.append("exit_not_after_entry")

    exp_sl = expected_stop_loss(entry_price, side)
    exp_tp = expected_take_profit(entry_price, side)

    checks["stop_loss_formula_ok"] = is_close(stop_loss, exp_sl)
    if not checks["stop_loss_formula_ok"]:
        details.append("stop_loss_formula_mismatch")

    checks["take_profit_formula_ok"] = is_close(take_profit, exp_tp)
    if not checks["take_profit_formula_ok"]:
        details.append("take_profit_formula_mismatch")

    exp_pnl = expected_pnl(entry_price, exit_price, volume, side)

    checks["pnl_formula_ok"] = np.isclose(pnl, exp_pnl, atol=PNL_ATOL, rtol=0.0)
    if not checks["pnl_formula_ok"]:
        details.append("pnl_formula_mismatch")

    if entry_bar is not None:
        checks["entry_price_matches_bar_open"] = is_close(entry_price, float(entry_bar["open"]))
        if not checks["entry_price_matches_bar_open"]:
            details.append("entry_price_not_bar_open")
    else:
        checks["entry_price_matches_bar_open"] = False

    if exit_bar is not None:
        exit_open = float(exit_bar["open"])
        exit_high = float(exit_bar["high"])
        exit_low = float(exit_bar["low"])
        exit_close = float(exit_bar["close"])

        if exit_reason == "stop_loss":
            checks["exit_price_matches_reason"] = is_close(exit_price, stop_loss)
            if side == "buy":
                checks["exit_bar_hit_level"] = exit_low <= stop_loss + PRICE_ATOL
            else:
                checks["exit_bar_hit_level"] = exit_high >= stop_loss - PRICE_ATOL

        elif exit_reason == "take_profit":
            checks["exit_price_matches_reason"] = is_close(exit_price, take_profit)
            if side == "buy":
                checks["exit_bar_hit_level"] = exit_high >= take_profit - PRICE_ATOL
            else:
                checks["exit_bar_hit_level"] = exit_low <= take_profit + PRICE_ATOL

        elif exit_reason == "max_hold":
            checks["exit_price_matches_reason"] = is_close(exit_price, exit_open)
            age_minutes = (exit_time - entry_time).total_seconds() / 60.0
            checks["exit_bar_hit_level"] = age_minutes >= MAX_HOLD_MINUTES

        elif exit_reason == "end_of_year":
            checks["exit_price_matches_reason"] = is_close(exit_price, exit_close)
            checks["exit_bar_hit_level"] = True

        else:
            checks["exit_price_matches_reason"] = False
            checks["exit_bar_hit_level"] = False
            details.append(f"unknown_exit_reason:{exit_reason}")

        if not checks["exit_price_matches_reason"]:
            details.append("exit_price_reason_mismatch")
        if not checks["exit_bar_hit_level"]:
            details.append("exit_bar_did_not_hit_level")
    else:
        checks["exit_price_matches_reason"] = False
        checks["exit_bar_hit_level"] = False

    all_ok = all(checks.values())

    return {
        "source_year": row["source_year"],
        "account_name": row["account_name"],
        "symbol": symbol,
        "side": side,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "volume": volume,
        "pnl": pnl,
        "expected_pnl": exp_pnl,
        **checks,
        "all_checks_passed": all_ok,
        "details": ";".join(details),
    }


def load_summaries_for_years(years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for year in years:
        summary_file = DATA_DIR / f"output_backtest_{year}" / "summary_backtest.csv"
        if not summary_file.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_file}")

        df = pd.read_csv(summary_file)
        if df.empty:
            continue

        df["source_year"] = year
        frames.append(df)

    if not frames:
        raise ValueError("No summary data loaded from the requested years.")

    return pd.concat(frames, ignore_index=True)


def audit_summary(trades: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trades.groupby(["source_year", "account_name"], as_index=False)
        .agg(
            recomputed_closed_trades=("pnl", "size"),
            recomputed_wins=("pnl", lambda s: int((s > 0).sum())),
            recomputed_losses=("pnl", lambda s: int((s < 0).sum())),
            recomputed_zero_pnl_trades=("pnl", lambda s: int(np.isclose(s, 0.0, atol=PNL_ATOL, rtol=0.0).sum())),
            recomputed_total_pnl=("pnl", "sum"),
        )
    )

    merged = summaries.merge(
        grouped,
        on=["source_year", "account_name"],
        how="left",
    )

    merged["recomputed_closed_trades"] = merged["recomputed_closed_trades"].fillna(0).astype(int)
    merged["recomputed_wins"] = merged["recomputed_wins"].fillna(0).astype(int)
    merged["recomputed_losses"] = merged["recomputed_losses"].fillna(0).astype(int)
    merged["recomputed_zero_pnl_trades"] = merged["recomputed_zero_pnl_trades"].fillna(0).astype(int)
    merged["recomputed_total_pnl"] = merged["recomputed_total_pnl"].fillna(0.0)

    merged["closed_trades_match"] = (
        merged["closed_trades"].astype(int) == merged["recomputed_closed_trades"]
    )
    merged["wins_match"] = (
        merged["wins"].astype(int) == merged["recomputed_wins"]
    )
    merged["losses_match"] = (
        merged["losses"].astype(int) == merged["recomputed_losses"]
    )
    merged["zero_pnl_count_match"] = (
        merged["closed_trades"].astype(int) ==
        (
            merged["recomputed_wins"] +
            merged["recomputed_losses"] +
            merged["recomputed_zero_pnl_trades"]
        )
    )
    merged["total_pnl_match"] = np.isclose(
        merged["total_pnl"].astype(float),
        merged["recomputed_total_pnl"].astype(float),
        atol=PNL_ATOL,
        rtol=0.0,
    )

    merged["all_summary_checks_passed"] = (
        merged["closed_trades_match"] &
        merged["wins_match"] &
        merged["losses_match"] &
        merged["zero_pnl_count_match"] &
        merged["total_pnl_match"]
    )

    return merged


def main() -> None:
    trades = load_trades_for_years(YEARS)
    if trades.empty:
        raise ValueError("No trades found in the requested backtest output folders.")

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="raise")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="raise")
    trades["symbol"] = trades["symbol"].astype(str)

    prices = load_prices_for_years(YEARS)
    price_lookup = prices.set_index(["symbol", "time"]).sort_index()

    audit_rows = [audit_trade(row, price_lookup) for _, row in trades.iterrows()]
    audit_df = pd.DataFrame(audit_rows)

    summaries = load_summaries_for_years(YEARS)
    summary_audit_df = audit_summary(trades, summaries)

    failed_df = audit_df[~audit_df["all_checks_passed"]].copy()
    summary_failed_df = summary_audit_df[~summary_audit_df["all_summary_checks_passed"]].copy()

    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not failed_df.empty:
        failed_df.to_csv(FAILED_AUDIT_FILE, index=False)
    elif FAILED_AUDIT_FILE.exists():
        FAILED_AUDIT_FILE.unlink()

    if not summary_failed_df.empty:
        summary_failed_df.to_csv(SUMMARY_AUDIT_FAILED_FILE, index=False)
    elif SUMMARY_AUDIT_FAILED_FILE.exists():
        SUMMARY_AUDIT_FAILED_FILE.unlink()

    print("=== TRADE AUDIT SUMMARY ===")
    print(f"Years covered: {YEARS}")
    print(f"Trades checked: {len(audit_df)}")
    print(f"Passed all checks: {int(audit_df['all_checks_passed'].sum())}")
    print(f"Failed any check: {len(failed_df)}")
    print()

    summary_cols = [
        "entry_bar_exists",
        "exit_bar_exists",
        "exit_after_entry",
        "stop_loss_formula_ok",
        "take_profit_formula_ok",
        "pnl_formula_ok",
        "entry_price_matches_bar_open",
        "exit_price_matches_reason",
        "exit_bar_hit_level",
    ]

    for col in summary_cols:
        passed = int(audit_df[col].sum())
        failed = len(audit_df) - passed
        print(f"{col}: passed={passed}, failed={failed}")

    print()
    print("=== SUMMARY FILE AUDIT ===")
    print(f"Summary rows checked: {len(summary_audit_df)}")
    print(f"Passed all summary checks: {int(summary_audit_df['all_summary_checks_passed'].sum())}")
    print(f"Failed any summary check: {len(summary_failed_df)}")
    print()

    for col in [
        "closed_trades_match",
        "wins_match",
        "losses_match",
        "zero_pnl_count_match",
        "total_pnl_match",
    ]:
        passed = int(summary_audit_df[col].sum())
        failed = len(summary_audit_df) - passed
        print(f"{col}: passed={passed}, failed={failed}")

    print()
    if failed_df.empty:
        print("No failed trade-audit file created.")
    else:
        print(f"Failed-only trade audit saved to: {FAILED_AUDIT_FILE}")

    if summary_failed_df.empty:
        print("No failed summary-audit file created.")
    else:
        print(f"Failed-only summary audit saved to: {SUMMARY_AUDIT_FAILED_FILE}")


if __name__ == "__main__":
    main()
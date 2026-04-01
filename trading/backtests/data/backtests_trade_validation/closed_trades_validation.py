from pathlib import Path
import pandas as pd
import numpy as np

# Years whose backtest outputs and price files will be audited together.
YEARS = [2022, 2023, 2024]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PRICES_DIR = DATA_DIR / "cleaned_data"

AUDIT_OUTPUT_DIR = SCRIPT_DIR
AUDIT_FILE = AUDIT_OUTPUT_DIR / "trade_audit_2022_2024.csv"
FAILED_AUDIT_FILE = AUDIT_OUTPUT_DIR / "trade_audit_2022_2024_failed.csv"

# Strategy constants used by the audit.
# These MUST match the constants used when the backtest itself was run.
STOP_LOSS_PCT = 0.003
TAKE_PROFIT_PCT = 0.006
MAX_HOLD_MINUTES = 120

# Small tolerances for floating-point comparisons.
# These prevent false failures caused by tiny rounding differences.
PRICE_ATOL = 1e-8
PNL_ATOL = 1e-6


def is_close(a: float, b: float, atol: float = PRICE_ATOL) -> bool:
    """
    Compare two numbers with a small absolute tolerance.

    We use this instead of == because floating-point values such as prices
    can differ by tiny rounding amounts even when they are effectively equal.
    """
    return np.isclose(float(a), float(b), atol=atol, rtol=0.0)


def expected_stop_loss(entry_price: float, side: str) -> float:
    """
    Recompute the stop-loss level from the entry price.

    This check answers:
    "Was the stop-loss stored in the trade log calculated correctly?"
    """
    if side == "buy":
        return entry_price * (1.0 - STOP_LOSS_PCT)
    if side == "sell":
        return entry_price * (1.0 + STOP_LOSS_PCT)
    raise ValueError(f"Unsupported side: {side}")


def expected_take_profit(entry_price: float, side: str) -> float:
    """
    Recompute the take-profit level from the entry price.

    This check answers:
    "Was the take-profit stored in the trade log calculated correctly?"
    """
    if side == "buy":
        return entry_price * (1.0 + TAKE_PROFIT_PCT)
    if side == "sell":
        return entry_price * (1.0 - TAKE_PROFIT_PCT)
    raise ValueError(f"Unsupported side: {side}")


def expected_pnl(entry_price: float, exit_price: float, volume: float, side: str) -> float:
    """
    Recompute the trade PnL from entry price, exit price, side, and volume.

    This check answers:
    "Is the logged realized PnL internally consistent with the trade prices?"
    """
    if side == "buy":
        return (exit_price - entry_price) * volume
    if side == "sell":
        return (entry_price - exit_price) * volume
    raise ValueError(f"Unsupported side: {side}")


def initial_risk_money(entry_price: float, stop_loss: float, volume: float, side: str) -> float:
    """
    Compute the initial money risk of the trade.

    This is used to derive the trade result in R-multiples:
        R = pnl / initial_risk_money

    This is not a pass/fail check. It is a useful extra diagnostic output.
    """
    if side == "buy":
        return (entry_price - stop_loss) * volume
    if side == "sell":
        return (stop_loss - entry_price) * volume
    raise ValueError(f"Unsupported side: {side}")


def load_trades_for_years(years: list[int]) -> pd.DataFrame:
    """
    Load closed-trade logs from multiple yearly backtest output folders.

    Expected input folders:
        data/backtest_output_2022/
        data/backtest_output_2023/
        data/backtest_output_2024/

    This step answers:
    "Do we have all trade ledgers needed for the years we want to audit?"
    """
    frames: list[pd.DataFrame] = []

    for year in years:
        trades_file = DATA_DIR / f"backtest_output_{year}" / "closed_trades_backtest.csv"
        if not trades_file.exists():
            raise FileNotFoundError(f"Missing trades file: {trades_file}")

        df = pd.read_csv(trades_file)
        if df.empty:
            continue

        # Keep track of which year each trade originally came from.
        df["source_year"] = year
        frames.append(df)

    if not frames:
        raise ValueError("No trade data loaded from the requested years.")

    trades = pd.concat(frames, ignore_index=True)
    return trades


def load_prices_for_years(years: list[int]) -> pd.DataFrame:
    """
    Load cleaned price data from multiple years and combine it into one table.

    This step answers:
    "Do we have all price bars needed to verify every trade entry and exit?"
    """
    frames: list[pd.DataFrame] = []

    for year in years:
        path = PRICES_DIR / f"cleaned_prices_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing price file: {path}")

        df = pd.read_csv(path)
        if df.empty:
            continue

        # Convert timestamps to timezone-aware UTC so they line up with trade timestamps.
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
        df["symbol"] = df["symbol"].astype(str)
        df["source_year"] = year
        frames.append(df)

    if not frames:
        raise ValueError("No price data loaded from the requested years.")

    prices = pd.concat(frames, ignore_index=True)

    # Sort and remove duplicate symbol+time rows if any exist.
    prices = prices.sort_values(["symbol", "time"]).drop_duplicates(
        subset=["symbol", "time"],
        keep="first"
    )
    prices = prices.reset_index(drop=True)
    return prices


def get_bar(price_lookup: pd.DataFrame, symbol: str, ts: pd.Timestamp) -> pd.Series | None:
    """
    Look up one exact price bar by (symbol, timestamp).

    This supports several later checks:
    - Did the entry bar exist?
    - Did the exit bar exist?
    - Did the entry price equal the bar open?
    - Did the exit bar actually hit the stop/target level?
    """
    try:
        row = price_lookup.loc[(symbol, ts)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    except KeyError:
        return None


def audit_trade(row: pd.Series, price_lookup: pd.DataFrame) -> dict:
    """
    Audit one closed trade against the stored strategy rules and historical bars.

    Checks performed:
    1. entry_bar_exists: Was there actually a bar at the logged entry timestamp?
    2. exit_bar_exists: Was there actually a bar at the logged exit timestamp?
    3. exit_after_entry: Did the trade close after it opened?
    4. stop_loss_formula_ok: Does the stored stop-loss match the configured stop-loss percentage?
    5. take_profit_formula_ok: Does the stored take-profit match the configured take-profit percentage?
    6. pnl_formula_ok: Does the stored PnL match the arithmetic from prices and volume?
    7. entry_price_matches_bar_open
       Does the trade entry price equal the open of the entry bar?
       This is what we expect because the backtest enters on the next M5 bar open.
    8. exit_price_matches_reason: Does the exit price equal the level implied by the exit reason?
       Examples:
       - stop_loss  -> exit_price should equal stop_loss
       - take_profit -> exit_price should equal take_profit
       - max_hold -> exit_price should equal exit bar open
    9. exit_bar_hit_level: Did the exit bar's OHLC actually touch the required exit level?
       Examples:
       - a long TP requires bar high >= take_profit
       - a short SL requires bar high >= stop_loss
       - max_hold requires the position age to be at least MAX_HOLD_MINUTES

    These are useful diagnostics even when all checks pass.
    """
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

    # Look up the exact bars used for entry and exit validation.
    entry_bar = get_bar(price_lookup, symbol, entry_time)
    exit_bar = get_bar(price_lookup, symbol, exit_time)

    checks: dict[str, bool] = {}
    details: list[str] = []

    # Check 1: the entry timestamp must exist in the price data.
    checks["entry_bar_exists"] = entry_bar is not None
    if not checks["entry_bar_exists"]:
        details.append("missing_entry_bar")

    # Check 2: the exit timestamp must exist in the price data.
    checks["exit_bar_exists"] = exit_bar is not None
    if not checks["exit_bar_exists"]:
        details.append("missing_exit_bar")

    # Check 3: exit must happen after entry.
    checks["exit_after_entry"] = exit_time > entry_time
    if not checks["exit_after_entry"]:
        details.append("exit_not_after_entry")

    # Recompute the theoretical SL/TP from the entry price.
    exp_sl = expected_stop_loss(entry_price, side)
    exp_tp = expected_take_profit(entry_price, side)

    # Check 4: stored SL matches formula.
    checks["stop_loss_formula_ok"] = is_close(stop_loss, exp_sl)
    if not checks["stop_loss_formula_ok"]:
        details.append("stop_loss_formula_mismatch")

    # Check 5: stored TP matches formula.
    checks["take_profit_formula_ok"] = is_close(take_profit, exp_tp)
    if not checks["take_profit_formula_ok"]:
        details.append("take_profit_formula_mismatch")

    # Recompute the trade PnL from prices and volume.
    exp_pnl = expected_pnl(entry_price, exit_price, volume, side)

    # Check 6: stored PnL matches formula.
    checks["pnl_formula_ok"] = np.isclose(pnl, exp_pnl, atol=PNL_ATOL, rtol=0.0)
    if not checks["pnl_formula_ok"]:
        details.append("pnl_formula_mismatch")

    # Check 7: entry price should equal entry bar open.
    if entry_bar is not None:
        checks["entry_price_matches_bar_open"] = is_close(entry_price, float(entry_bar["open"]))
        if not checks["entry_price_matches_bar_open"]:
            details.append("entry_price_not_bar_open")
    else:
        checks["entry_price_matches_bar_open"] = False

    # Checks 8 and 9 depend on the exit reason and the OHLC of the exit bar.
    if exit_bar is not None:
        exit_open = float(exit_bar["open"])
        exit_high = float(exit_bar["high"])
        exit_low = float(exit_bar["low"])

        if exit_reason == "stop_loss":
            # Check 8: if exit reason is stop-loss, the exit price should equal stop_loss.
            checks["exit_price_matches_reason"] = is_close(exit_price, stop_loss)

            # Check 9: the exit bar must actually hit the stop-loss level.
            if side == "buy":
                checks["exit_bar_hit_level"] = exit_low <= stop_loss + PRICE_ATOL
            else:
                checks["exit_bar_hit_level"] = exit_high >= stop_loss - PRICE_ATOL

        elif exit_reason == "take_profit":
            # Check 8: if exit reason is take-profit, the exit price should equal take_profit.
            checks["exit_price_matches_reason"] = is_close(exit_price, take_profit)

            # Check 9: the exit bar must actually hit the take-profit level.
            if side == "buy":
                checks["exit_bar_hit_level"] = exit_high >= take_profit - PRICE_ATOL
            else:
                checks["exit_bar_hit_level"] = exit_low <= take_profit + PRICE_ATOL

        elif exit_reason == "max_hold":
            # Check 8: if exit reason is max-hold, the exit price should equal exit bar open.
            checks["exit_price_matches_reason"] = is_close(exit_price, exit_open)

            # Check 9: trade age must be at least the configured maximum hold time.
            age_minutes = (exit_time - entry_time).total_seconds() / 60.0
            checks["exit_bar_hit_level"] = age_minutes >= MAX_HOLD_MINUTES

        else:
            # Unknown exit reasons automatically fail both exit checks.
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

    # A trade passes only if every individual check passed.
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


def main() -> None:
    """
    Main flow of the audit script.

    1. Load all yearly closed-trade ledgers.
    2. Parse trade timestamps.
    3. Load all yearly cleaned price files.
    4. Build a fast symbol+time lookup for bars.
    5. Audit every trade.
    6. Save:
       - one full audit file
       - one failed-only audit file
    7. Print a summary of how many trades passed/failed each check.
    """
    trades = load_trades_for_years(YEARS)
    if trades.empty:
        raise ValueError("No trades found in the requested backtest output folders.")

    # Parse trade timestamps as UTC so they align exactly with the price data.
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="raise")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="raise")
    trades["symbol"] = trades["symbol"].astype(str)

    prices = load_prices_for_years(YEARS)

    # Fast lookup indexed by (symbol, time) for exact bar matching.
    price_lookup = prices.set_index(["symbol", "time"]).sort_index()

    # Audit each trade row one by one.
    audit_rows = [audit_trade(row, price_lookup) for _, row in trades.iterrows()]
    audit_df = pd.DataFrame(audit_rows)

    # Save full audit and failed-only subset.
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_FILE, index=False)

    failed_df = audit_df[~audit_df["all_checks_passed"]].copy()
    failed_df.to_csv(FAILED_AUDIT_FILE, index=False)

    print("=== TRADE AUDIT SUMMARY ===")
    print(f"Years covered: {YEARS}")
    print(f"Trades checked: {len(audit_df)}")
    print(f"Passed all checks: {int(audit_df['all_checks_passed'].sum())}")
    print(f"Failed any check: {len(failed_df)}")
    print()

    # Print a pass/fail count for each individual check
    # so it is easy to see where problems occur.
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
    print(f"Full audit saved to: {AUDIT_FILE}")
    print(f"Failed-only audit saved to: {FAILED_AUDIT_FILE}")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from pathlib import Path

# Set the backtest year here
YEAR = 2022

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / f"output_backtest_{YEAR}"
TRADES_FILE = INPUT_DIR / "closed_trades_backtest.csv"
EQUITY_FILE = INPUT_DIR / "equity_curve_backtest.csv"
OUTPUT_FILE = BASE_DIR / f"performance_metrics_{YEAR}.csv"


def initial_risk_money(row: pd.Series) -> float:
    """
    Compute the INITIAL money risk of one trade.

    This is the denominator used to convert a realized trade result into R-units.

    For a buy:
        risk_money = (entry_price - stop_loss) * volume

    For a sell:
        risk_money = (stop_loss - entry_price) * volume

    If a trade hits stop loss exactly, then:
        pnl / risk_money ≈ -1R
    """
    if row["side"] == "buy":
        return (row["entry_price"] - row["stop_loss"]) * row["volume"]
    elif row["side"] == "sell":
        return (row["stop_loss"] - row["entry_price"]) * row["volume"]
    return np.nan


def compute_sigma_R(R_values: pd.Series, expectancy_R: float) -> float:
    """
    Compute sigma_R exactly as in the screenshot:

        sigma_R = sqrt( (1/N) * sum((R_i - E[R])^2) )

    This uses the population-style denominator N, matching the slide.
    """
    N = len(R_values)
    if N == 0:
        return np.nan

    variance = ((R_values - expectancy_R) ** 2).sum() / N
    return float(np.sqrt(variance))


def compute_max_drawdown(equity_series: pd.Series) -> float:
    """
    Compute maximum drawdown:

        MaxDD = max_t ((Peak_t - Equity_t) / Peak_t)

    We compute the running peak first, then the drawdown at each timestamp.
    """
    if len(equity_series) == 0:
        return np.nan

    running_peak = equity_series.cummax()
    drawdowns = (running_peak - equity_series) / running_peak
    return float(drawdowns.max())


def main() -> None:
    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Could not find trades file: {TRADES_FILE}")

    if not EQUITY_FILE.exists():
        raise FileNotFoundError(f"Could not find equity file: {EQUITY_FILE}")

    # Load the closed-trade ledger for the selected year
    trades_df = pd.read_csv(TRADES_FILE)

    # Load the full equity curve for the selected year
    equity_df = pd.read_csv(EQUITY_FILE)

    # Parse equity timestamps so each account can be sorted correctly in time
    equity_df["time"] = pd.to_datetime(equity_df["time"], utc=True, errors="raise")

    # Compute initial money risk for every trade
    trades_df["risk_money"] = trades_df.apply(initial_risk_money, axis=1)

    # Convert each realized trade result into an R-multiple:
    #
    #   R_i = pnl_i / initial_risk_money_i
    #
    # - full stop loss hit -> about -1R
    # - full take profit hit with 0.3% SL and 0.6% TP -> about +2R
    trades_df["R_i"] = trades_df["pnl"] / trades_df["risk_money"]

    results = []

    # Compute all metrics separately for each account / strategy variant
    for account_name, g in trades_df.groupby("account_name"):
        # Winning trades: R_i > 0
        wins = g[g["R_i"] > 0]

        # Losing trades: R_i < 0
        losses = g[g["R_i"] < 0]

        # Counts used in the formulas
        N_trades = len(g)
        N_wins = len(wins)
        N_losses = len(losses)

        # Formula:
        #
        #   p_w = N_wins / N_trades
        #
        win_rate = N_wins / N_trades if N_trades > 0 else 0.0

        # Formula:
        #
        #   \bar{R}_w = average winning trade in R-units
        #   \bar{R}_l = average losing trade in R-units, expressed as POSITIVE magnitude
        #
        R_w_bar = wins["R_i"].sum() / N_wins if N_wins > 0 else 0.0
        R_l_bar = abs(losses["R_i"].sum() / N_losses) if N_losses > 0 else 0.0

        # Formula:
        #
        #   E[R] = p_w * \bar{R}_w - (1 - p_w) * \bar{R}_l
        #
        expectancy_R = win_rate * R_w_bar - (1 - win_rate) * R_l_bar

        # Formula:
        #
        #   PF = sum(R_i^+) / |sum(R_i^-)|
        #
        gross_profit_R = wins["R_i"].sum()
        gross_loss_R = abs(losses["R_i"].sum())
        profit_factor = gross_profit_R / gross_loss_R if gross_loss_R > 0 else np.nan

        # Formula:
        #
        #   sigma_R = sqrt( (1/N) * sum((R_i - E[R])^2) )
        #
        sigma_R = compute_sigma_R(g["R_i"], expectancy_R)

        # Formula:
        #
        #   Sharpe = E[R] / sigma_R
        #
        sharpe = expectancy_R / sigma_R if pd.notna(sigma_R) and sigma_R > 0 else np.nan

        # Formula:
        #
        #   MaxDD = max_t ((Peak_t - Equity_t) / Peak_t)
        #
        # This metric must come from the equity curve, using the full yearly path.
        g_equity = equity_df[equity_df["account_name"] == account_name].copy()
        g_equity = g_equity.sort_values("time")

        max_drawdown = np.nan
        if not g_equity.empty:
            equity_series = g_equity["equity"].astype(float)
            max_drawdown = compute_max_drawdown(equity_series)

        results.append({
            "year": YEAR,
            "account_name": account_name,
            "win_rate": win_rate,
            "R_w_bar": R_w_bar,
            "R_l_bar": R_l_bar,
            "expectancy_R": expectancy_R,
            "profit_factor": profit_factor,
            "sigma_R": sigma_R,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        })

    results_df = pd.DataFrame(results)

    # Print the final metrics per account for quick inspection
    print(results_df)

    # Save one performance file for this year in the script folder
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved metrics to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
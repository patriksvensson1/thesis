from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

BASE_DIR = Path(__file__).resolve().parent

REPORT_SPECS = [
    {
        "file": BASE_DIR / "ReportHistory-52827321.html",
        "account_label": "non_decayed_account",
    },
    {
        "file": BASE_DIR / "ReportHistory-52827323.html",
        "account_label": "decayed_account",
    },
]

OUTPUT_FILE = BASE_DIR / "performance_forward_tests.csv"


def parse_mt5_number(value: str) -> float:
    """
    Convert MT5-style numeric text into float.

    Examples:
        '451.3'
        '-476.75'
        '8 251.13'
        '2.02%'
    """
    if value is None:
        return np.nan

    text = str(value).strip()
    text = text.replace("\xa0", "").replace(" ", "").replace("%", "")

    if text == "":
        return np.nan

    return float(text)


def initial_risk_money(row: pd.Series) -> float:
    """
    Compute the initial money risk of one trade.

    Formula:
        For a buy:
            risk_money = (entry_price - stop_loss) * volume

        For a sell:
            risk_money = (stop_loss - entry_price) * volume
    """
    if row["side"] == "buy":
        return (row["entry_price"] - row["stop_loss"]) * row["volume"]
    elif row["side"] == "sell":
        return (row["stop_loss"] - row["entry_price"]) * row["volume"]
    return np.nan


def compute_sigma_R(R_values: pd.Series, expectancy_R: float) -> float:
    """
    Compute sigma_R exactly as in the thesis formula:

        sigma_R = sqrt( (1/N) * sum((R_i - E[R])^2) )
    """
    N = len(R_values)
    if N == 0:
        return np.nan

    variance = ((R_values - expectancy_R) ** 2).sum() / N
    return float(np.sqrt(variance))


def extract_account_number_from_report(soup: BeautifulSoup) -> str:
    """
    Extract numeric account id from the MT5 HTML report header.
    """
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Account:\s*([0-9]+)", text)
    if match:
        return match.group(1)

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    match = re.match(r"([0-9]+):", title)
    if match:
        return match.group(1)

    return "unknown_account"


def extract_max_drawdown_from_report(html_text: str) -> float:
    """
    Extract Balance Drawdown Relative from the MT5 report summary.

    Returns decimal form:
        2.02% -> 0.0202
    """
    match = re.search(
        r"Balance Drawdown Relative:</td>\s*<td[^>]*><b>\s*([0-9]+(?:\.[0-9]+)?)%",
        html_text,
        flags=re.IGNORECASE,
    )

    if not match:
        return np.nan

    return float(match.group(1)) / 100.0


def extract_positions_table(soup: BeautifulSoup) -> pd.DataFrame:
    """
    Parse the Positions section from the MT5 HTML report.

    Each data row should contain:
        open_time, position_id, symbol, side, volume, entry_price,
        stop_loss, take_profit, close_time, close_price,
        commission, swap, profit
    """
    rows = soup.find_all("tr")

    in_positions_section = False
    data_rows = []

    for tr in rows:
        header_div = tr.find("div")
        if header_div:
            section_name = header_div.get_text(" ", strip=True)
            if section_name == "Positions":
                in_positions_section = True
                continue
            if in_positions_section and section_name != "Positions":
                break

        if not in_positions_section:
            continue

        visible_cells = []
        for td in tr.find_all("td"):
            classes = td.get("class", [])
            if "hidden" in classes:
                continue
            visible_cells.append(td.get_text(" ", strip=True))

        if len(visible_cells) != 13:
            continue

        # Skip the header row inside the positions table
        if visible_cells[0] == "Time" and visible_cells[1] == "Position":
            continue

        try:
            pd.to_datetime(visible_cells[0], format="%Y.%m.%d %H:%M:%S")
            int(visible_cells[1])
        except Exception:
            continue

        data_rows.append({
            "open_time": visible_cells[0],
            "position_id": visible_cells[1],
            "symbol": visible_cells[2],
            "side": visible_cells[3].strip().lower(),
            "volume": visible_cells[4],
            "entry_price": visible_cells[5],
            "stop_loss": visible_cells[6],
            "take_profit": visible_cells[7],
            "close_time": visible_cells[8],
            "close_price": visible_cells[9],
            "pnl_commission": visible_cells[10],
            "pnl_swap": visible_cells[11],
            "pnl_profit": visible_cells[12],
        })

    if not data_rows:
        raise RuntimeError("Could not find any rows in the Positions section of the HTML report.")

    df = pd.DataFrame(data_rows)

    df["position_id"] = pd.to_numeric(df["position_id"], errors="raise").astype("int64")
    df["open_time"] = pd.to_datetime(
        df["open_time"], format="%Y.%m.%d %H:%M:%S", utc=True, errors="raise"
    )
    df["close_time"] = pd.to_datetime(
        df["close_time"], format="%Y.%m.%d %H:%M:%S", utc=True, errors="raise"
    )

    numeric_cols = [
        "volume",
        "entry_price",
        "stop_loss",
        "take_profit",
        "close_price",
        "pnl_commission",
        "pnl_swap",
        "pnl_profit",
    ]
    for col in numeric_cols:
        df[col] = df[col].apply(parse_mt5_number)

    df["pnl_fee"] = 0.0

    # net_result = realized trade result after costs
    df["net_result"] = (
        df["pnl_profit"]
        + df["pnl_commission"]
        + df["pnl_swap"]
        + df["pnl_fee"]
    )

    return df


def compute_metrics_for_report(report_file: Path, account_label: str) -> dict:
    if not report_file.exists():
        raise FileNotFoundError(f"Could not find HTML report: {report_file}")

    # MT5 exports this report as UTF-16
    html_text = report_file.read_text(encoding="utf-16", errors="strict")
    soup = BeautifulSoup(html_text, "html.parser")

    account_number = extract_account_number_from_report(soup)
    max_drawdown = extract_max_drawdown_from_report(html_text)

    trades_df = extract_positions_table(soup)
    trades_df["account_number"] = account_number
    trades_df["account_name"] = account_label

    # Basic sanity filtering
    trades_df = trades_df[trades_df["side"].isin(["buy", "sell"])].copy()
    trades_df = trades_df[pd.notna(trades_df["entry_price"])].copy()
    trades_df = trades_df[pd.notna(trades_df["stop_loss"])].copy()
    trades_df = trades_df[pd.notna(trades_df["volume"])].copy()
    trades_df = trades_df[pd.notna(trades_df["net_result"])].copy()

    trades_df["risk_money"] = trades_df.apply(initial_risk_money, axis=1)

    trades_df = trades_df[pd.notna(trades_df["risk_money"])].copy()
    trades_df = trades_df[trades_df["risk_money"] > 0].copy()

    # R_i = realized trade result / initial money risk
    trades_df["R_i"] = trades_df["net_result"] / trades_df["risk_money"]

    wins = trades_df[trades_df["R_i"] > 0]
    losses = trades_df[trades_df["R_i"] < 0]

    N_trades = len(trades_df)
    N_wins = len(wins)
    N_losses = len(losses)

    # Win rate
    win_rate = N_wins / N_trades if N_trades > 0 else 0.0

    # Average winning and losing R-multiples
    R_w_bar = wins["R_i"].sum() / N_wins if N_wins > 0 else 0.0
    R_l_bar = abs(losses["R_i"].sum() / N_losses) if N_losses > 0 else 0.0

    # Expectancy in risk units
    expectancy_R = win_rate * R_w_bar - (1 - win_rate) * R_l_bar

    # Profit factor from the R_i-series
    gross_profit_R = wins["R_i"].sum()
    gross_loss_R = abs(losses["R_i"].sum())
    profit_factor = gross_profit_R / gross_loss_R if gross_loss_R > 0 else np.nan

    # sigma_R and Sharpe ratio
    sigma_R = compute_sigma_R(trades_df["R_i"], expectancy_R)
    sharpe = expectancy_R / sigma_R if pd.notna(sigma_R) and sigma_R > 0 else np.nan

    return {
        "account_name": account_label,
        "account_number": account_number,
        "win_rate": win_rate,
        "R_w_bar": R_w_bar,
        "R_l_bar": R_l_bar,
        "expectancy_R": expectancy_R,
        "profit_factor": profit_factor,
        "sigma_R": sigma_R,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "N_trades": N_trades,
        "N_wins": N_wins,
        "N_losses": N_losses,
    }


def main() -> None:
    results = []

    for spec in REPORT_SPECS:
        result = compute_metrics_for_report(
            report_file=spec["file"],
            account_label=spec["account_label"],
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print(results_df)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved metrics to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
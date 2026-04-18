# Thesis

This project was undertaken and completed as a thesis for the DVK Uppsats program VT 2026 at Stockholm University. Its primary objective was to compare two otherwise identical algorithmic trading strategies that differed only in how news sentiment was aggregated over time: one using equal-weight sentiment aggregation and one using time-decay-weighted aggregation. The system combines financial news from GDELT, sentiment classification with FinBERT, and price-based prediction using an LSTM model.

## Forward testing

To reproduce the forward-test setup, valid MetaTrader 5 demo-account credentials must be added in `config.py`. The repository is configured to run two accounts: one for the no-decay strategy and one for the decay-based strategy.

Once the account credentials have been added, the forward-test workflow can be started by executing `main.py`.

Data generated during forward testing, including logs produced by the run, are written to the `data/` folder.

## Backtesting

The cleaned historical data required for the backtests are already prepared in the repository. To reproduce a backtest, set the `YEAR` variable in `main.py` to `2022`, `2023`, or `2024`, and then execute `main.py`. The script will automatically load the corresponding cleaned price and news files and save the generated backtest outputs to a year-specific folder under `data/`.
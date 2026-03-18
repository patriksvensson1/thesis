import pandas as pd
import numpy as np

INPUT_FILE = "LSTM_Training/raw_price_data_M15.csv"
OUTPUT_FILE = "LSTM_Training/lstm_dataset_M15.csv"

# 15-minute bars
HORIZON = 8                 # predict 8 bars ahead = 120 minutes
UP_THRESHOLD = 0.003        # +0.30%
DOWN_THRESHOLD = -0.003     # -0.30%

df = pd.read_csv(INPUT_FILE)
df["time"] = pd.to_datetime(df["time"], utc=True)

required_cols = ["time", "symbol", "open", "high", "low", "close", "tick_volume"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = (
    df.sort_values(["symbol", "time"])
      .drop_duplicates(subset=["symbol", "time"])
      .reset_index(drop=True)
)

def add_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g["symbol"] = group.name

    # basic return / bar-shape features
    g["return_1"] = g["close"].pct_change(1)
    g["return_3"] = g["close"].pct_change(3)
    g["return_6"] = g["close"].pct_change(6)
    g["return_12"] = g["close"].pct_change(12)

    g["oc_change"] = (g["close"] - g["open"]) / g["open"].replace(0, np.nan)
    g["hl_range"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)

    g["volume_change_1"] = g["tick_volume"].replace(0, np.nan).pct_change(1)
    g["volatility_6"] = g["return_1"].rolling(6).std()
    g["volatility_20"] = g["return_1"].rolling(20).std()

    # regime / trend features
    g["ma_5"] = g["close"].rolling(5).mean()
    g["ma_20"] = g["close"].rolling(20).mean()
    g["ma_gap"] = (g["ma_5"] - g["ma_20"]) / g["ma_20"]

    g["rolling_high_20"] = g["high"].rolling(20).max()
    g["rolling_low_20"] = g["low"].rolling(20).min()
    g["range_position_20"] = (
        (g["close"] - g["rolling_low_20"]) /
        (g["rolling_high_20"] - g["rolling_low_20"])
    )

    g["avg_volume_20"] = g["tick_volume"].rolling(20).mean()
    g["volume_ratio_20"] = g["tick_volume"] / g["avg_volume_20"]

    # future target over chosen horizon
    g["future_close"] = g["close"].shift(-HORIZON)
    g["future_return_h"] = g["future_close"] / g["close"] - 1.0

    # 3-zone labeling: up / down / neutral
    g["target"] = np.nan
    g.loc[g["future_return_h"] > UP_THRESHOLD, "target"] = 1
    g.loc[g["future_return_h"] < DOWN_THRESHOLD, "target"] = 0

    return g

df = df.groupby("symbol", group_keys=False).apply(add_features).reset_index(drop=True)

feature_cols = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "oc_change",
    "hl_range",
    "volume_change_1",
    "volatility_6",
    "volatility_20",
    "ma_gap",
    "range_position_20",
    "volume_ratio_20",
]

needed_cols = feature_cols + [
    "future_close",
    "future_return_h",
    "target",
]

# replace inf / -inf with NaN
df[needed_cols] = df[needed_cols].replace([np.inf, -np.inf], np.nan)

print("NaN counts before drop:")
print(df[needed_cols].isna().sum())

# drop rows with missing features, future values, or neutral target
df = df.dropna(subset=needed_cols).reset_index(drop=True)

# target should now only be 0 or 1
df["target"] = df["target"].astype(int)

# validate target
expected_target = np.where(df["future_return_h"] > UP_THRESHOLD, 1, 0)
if not (df["target"] == expected_target).all():
    bad_rows = df.loc[
        df["target"] != expected_target,
        ["symbol", "time", "close", "future_close", "future_return_h", "target"]
    ]
    raise ValueError(f"Target validation failed. Example bad rows:\n{bad_rows.head(10)}")

# validate future return
expected_future_return = df["future_close"] / df["close"] - 1.0
if not np.allclose(df["future_return_h"], expected_future_return, equal_nan=False):
    bad_mask = ~np.isclose(df["future_return_h"], expected_future_return, equal_nan=False)
    bad_rows = df.loc[
        bad_mask,
        ["symbol", "time", "close", "future_close", "future_return_h"]
    ]
    raise ValueError(f"future_return_h validation failed. Example bad rows:\n{bad_rows.head(10)}")

output_cols = [
    "time",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
] + feature_cols + ["future_return_h", "target"]

df = df[output_cols]
df.to_csv(OUTPUT_FILE, index=False)

print("Preprocessing completed successfully.")
print(f"Saved {OUTPUT_FILE} with {len(df)} rows.")
print(f"Symbols: {df['symbol'].nunique()}")
print(f"Date range: {df['time'].min()} -> {df['time'].max()}")
print("Class balance:")
print(df["target"].value_counts(normalize=True))
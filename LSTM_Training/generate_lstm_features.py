import pandas as pd
import numpy as np

INPUT_FILE = "LSTM_Training/raw_price_data_M15.csv"
OUTPUT_FILE = "LSTM_Training/lstm_dataset_M15.csv"

df = pd.read_csv(INPUT_FILE)
df["time"] = pd.to_datetime(df["time"], utc=True)

# Basic cleaning
df = (
    df.sort_values(["symbol", "time"])
      .drop_duplicates(subset=["symbol", "time"])
      .reset_index(drop=True)
)

def add_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g["symbol"] = group.name

    # Input Features
    g["return_1"] = g["close"].pct_change(1)
    g["return_3"] = g["close"].pct_change(3)
    g["return_6"] = g["close"].pct_change(6)

    # Bar shape features
    g["oc_change"] = (g["close"] - g["open"]) / g["open"].replace(0, np.nan)
    g["hl_range"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)

    # Momentum/Volatility features
    g["volume_change_1"] = g["tick_volume"].replace(0, np.nan).pct_change(1)
    g["volatility_6"] = g["return_1"].rolling(6).std()

    # Target logic (8 bars ahead = 120 minutes)
    g["future_close"] = g["close"].shift(-8)
    g["future_return_h"] = g["future_close"] / g["close"] - 1.0

    g["target"] = np.nan
    g.loc[g["future_return_h"] > 0.003, "target"] = 1   # +0.30%
    g.loc[g["future_return_h"] < -0.003, "target"] = 0  # -0.30%

    return g

# Apply the simplified features
df = df.groupby("symbol", group_keys=False).apply(add_features).reset_index(drop=True)

# List of features used in the LSTM
feature_cols = [
    "return_1", "return_3", "return_6",
    "oc_change", "hl_range",
    "volume_change_1", "volatility_6"
]

# Columns needed for the cleanup process
needed_cols = feature_cols + ["future_return_h", "target"]

# Clean data
df[needed_cols] = df[needed_cols].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=needed_cols).reset_index(drop=True)
df["target"] = df["target"].astype(int)

# Final output selection
output_cols = ["time", "symbol"] + feature_cols + ["target"]
df = df[output_cols]

df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned dataset saved with {len(df)} rows and {len(feature_cols)} features.")
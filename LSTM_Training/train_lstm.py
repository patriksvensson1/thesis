import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

INPUT_FILE = "LSTM_Training/lstm_dataset_M15.csv"
OUTPUT_DIR = Path("trained_lstm")
OUTPUT_DIR.mkdir(exist_ok=True)

SEQUENCE_LENGTH = 12
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 20
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

df = pd.read_csv(INPUT_FILE)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values(["symbol", "time"]).reset_index(drop=True)

feature_cols = [
    "return_1",
    "return_3",
    "return_6",
    "oc_change",
    "hl_range",
    "volume_change_1",
    "volatility_6",
]

required_cols = ["time", "symbol", "target"] + feature_cols
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

all_times = np.sort(df["time"].unique())
n_times = len(all_times)

train_end_idx = int(n_times * TRAIN_RATIO)
val_end_idx = int(n_times * (TRAIN_RATIO + VAL_RATIO))

train_end_time = all_times[train_end_idx - 1]
val_end_time = all_times[val_end_idx - 1]

train_df = df[df["time"] <= train_end_time].copy()
val_df = df[(df["time"] > train_end_time) & (df["time"] <= val_end_time)].copy()
test_df = df[df["time"] > val_end_time].copy()

print("Train rows:", len(train_df))
print("Val rows:", len(val_df))
print("Test rows:", len(test_df))
print("Train end:", train_end_time)
print("Val end:", val_end_time)

scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_df[feature_cols] = scaler.transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

joblib.dump(scaler, OUTPUT_DIR / "feature_scaler.pkl")

def build_sequences(data: pd.DataFrame, seq_len: int, feature_columns: list[str]):
    X, y, meta = [], [], []

    for symbol, group in data.groupby("symbol"):
        g = group.sort_values("time").reset_index(drop=True)

        features = g[feature_columns].values.astype(np.float32)
        targets = g["target"].values.astype(np.float32)
        times = g["time"].values

        for i in range(seq_len, len(g)):
            X.append(features[i - seq_len:i])
            y.append(targets[i])
            meta.append((symbol, str(times[i])))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, meta

X_train, y_train, meta_train = build_sequences(train_df, SEQUENCE_LENGTH, feature_cols)
X_val, y_val, meta_val = build_sequences(val_df, SEQUENCE_LENGTH, feature_cols)
X_test, y_test, meta_test = build_sequences(test_df, SEQUENCE_LENGTH, feature_cols)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError("One of the datasets is empty after sequence building.")

model = Sequential([
    LSTM(64, input_shape=(SEQUENCE_LENGTH, len(feature_cols))),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

test_probs = model.predict(X_test, verbose=0).flatten()
test_preds = (test_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_probs)

print("\nTest Accuracy:", round(acc, 4))
print("Test AUC:", round(auc, 4))
print("\nClassification Report:")
print(classification_report(y_test, test_preds, digits=4))

model.save(OUTPUT_DIR / "lstm_direction_model.keras")

metadata = {
    "sequence_length": SEQUENCE_LENGTH,
    "feature_columns": feature_cols,
    "train_end_time": str(train_end_time),
    "val_end_time": str(val_end_time),
    "n_train_sequences": int(len(X_train)),
    "n_val_sequences": int(len(X_val)),
    "n_test_sequences": int(len(X_test)),
}

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved model, scaler, and metadata to: {OUTPUT_DIR.resolve()}")
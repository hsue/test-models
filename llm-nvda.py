"""
NVDA Next-Week Price Predictor — Neural Network (LSTM) Version
=================================================================
Same learning-exercise caveat as before: this demonstrates how a neural
network is built and trained, NOT a reliable trading signal. Short-term
stock prices are extremely hard to predict from price history alone.

Install requirements first:
    pip install yfinance scikit-learn pandas numpy torch
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# -----------------------------
# STEP 1: Get the data (same as before)
# -----------------------------
ticker = "NVDA"
df = yf.download(ticker, period="5y", interval="1d")
print(f"Downloaded {len(df)} days of {ticker} data")

# -----------------------------
# STEP 2: Feature engineering (same features as the Random Forest version)
# -----------------------------
df["return_1d"] = df["Close"].pct_change(1)
df["return_5d"] = df["Close"].pct_change(5)
df["return_10d"] = df["Close"].pct_change(10)
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["ma_ratio"] = df["ma_5"] / df["ma_20"]
df["volatility_10d"] = df["return_1d"].rolling(10).std()
df["volume_change"] = df["Volume"].pct_change(5)
df["target_5d_forward_return"] = df["Close"].shift(-5) / df["Close"] - 1
df = df.dropna()

feature_cols = ["return_1d", "return_5d", "return_10d", "ma_ratio", "volatility_10d", "volume_change"]

# -----------------------------
# STEP 3: Build sequences — this is what's actually different for a neural
# network. Instead of one row of features per prediction, an LSTM looks at
# a WINDOW of recent days in sequence, so it can learn patterns over time
# rather than just a single day's snapshot.
# -----------------------------
SEQUENCE_LENGTH = 20  # look back 20 trading days for each prediction

def build_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)

# Scale features — neural networks train much better when inputs are
# normalized to a similar range, unlike Random Forests which don't care.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_cols])
targets = df["target_5d_forward_return"].values

X, y = build_sequences(scaled_features, targets, SEQUENCE_LENGTH)

# Chronological split — same rule as before: never shuffle time series data
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"\nTraining on {len(X_train)} sequences, testing on {len(X_test)} sequences")

# -----------------------------
# STEP 4: Define the neural network architecture
# -----------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # take the final time step's output
        return self.fc(last_step)

model = LSTMPredictor(input_size=len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# -----------------------------
# STEP 5: Train the network
# -----------------------------
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = loss_fn(predictions, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Training loss: {loss.item():.6f}")

# -----------------------------
# STEP 6: Evaluate — same honesty check as before
# -----------------------------
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t).numpy().flatten()

mae = mean_absolute_error(y_test, test_predictions)
baseline_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))
direction_correct = np.mean(np.sign(test_predictions) == np.sign(y_test))

print(f"\nModel MAE: {mae:.4f} ({mae*100:.2f}%)")
print(f"Naive baseline (predict 0% change) MAE: {baseline_mae:.4f} ({baseline_mae*100:.2f}%)")
print(f"Directional accuracy: {direction_correct*100:.1f}% (50% = coin flip)")

# -----------------------------
# STEP 7: Predict next week using the most recent sequence
# -----------------------------
latest_sequence = scaled_features[-SEQUENCE_LENGTH:]
latest_sequence_t = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)

model.eval()
with torch.no_grad():
    next_week_predicted_return = model(latest_sequence_t).item()

current_price = df["Close"].iloc[-1]
predicted_price = current_price * (1 + next_week_predicted_return)

print(f"\nCurrent price: ${current_price:.2f}")
print(f"Model's predicted return over next ~5 trading days: {next_week_predicted_return*100:.2f}%")
print(f"Implied predicted price: ${predicted_price:.2f}")
print("\n⚠️  Same reminder as before: compare MAE and directional accuracy above")
print("    against the baseline before trusting this number for anything real.")
print("    A neural network is more flexible/powerful than a Random Forest, but")
print("    that also means it can more easily overfit to noise in a small,")
print("    noisy dataset like 5 years of daily stock prices.")
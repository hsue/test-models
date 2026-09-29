"""
NVDA Next-Month Price Predictor — ML Learning Exercise
=======================================================
IMPORTANT: This is for learning the ML workflow, NOT a real trading tool.
Stock prices (especially over a month-long horizon) are extremely difficult to
predict from historical price data alone — most academic research supports
something close to a "random walk," meaning next month's move is dominated by
news, earnings, and macro events the model has no way to see in advance.
Treat any output here as an educational exercise in a few important
techniques, not a forecast to act on. A longer horizon like this also means
MORE can happen in the meantime (earnings reports, product news, macro
shifts), which tends to make errors larger than the 1-week version.

Run this on your own machine or in Google Colab (needs internet access to
download stock data). Install requirements first:
    pip install yfinance scikit-learn pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# STEP 1: Get the data
# -----------------------------
# Download ~5 years of daily NVDA price history
ticker = "NVDA"
df = yf.download(ticker, period="5y", interval="1d")
print(f"Downloaded {len(df)} days of {ticker} data")
print(df.tail())

# -----------------------------
# STEP 2: Feature engineering
# -----------------------------
# We can't just feed raw prices to a model — we build "features" that might
# carry predictive signal: past returns, moving averages, volatility, volume trends.

# ~21 trading days = ~1 calendar month (markets are open ~21 days/month on average)
HORIZON_DAYS = 21

df["return_1d"] = df["Close"].pct_change(1)
df["return_5d"] = df["Close"].pct_change(5)
df["return_10d"] = df["Close"].pct_change(10)
df["return_20d"] = df["Close"].pct_change(20)  # added: longer lookback fits a longer horizon
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["ma_50"] = df["Close"].rolling(50).mean()  # added: a longer moving average for trend context
df["ma_ratio"] = df["ma_5"] / df["ma_20"]  # short vs medium trend
df["ma_ratio_long"] = df["ma_20"] / df["ma_50"]  # medium vs long trend
df["volatility_10d"] = df["return_1d"].rolling(10).std()
df["volatility_20d"] = df["return_1d"].rolling(20).std()  # added: volatility over a longer window
df["volume_change"] = df["Volume"].pct_change(5)

# TARGET: what we're trying to predict — the % price change HORIZON_DAYS trading
# days (roughly 1 month) into the future
df["target_forward_return"] = df["Close"].shift(-HORIZON_DAYS) / df["Close"] - 1

# Drop rows with missing values (created by rolling windows / shifting)
df = df.dropna()

feature_cols = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "ma_ratio", "ma_ratio_long", "volatility_10d", "volatility_20d", "volume_change",
]
X = df[feature_cols]
y = df["target_forward_return"]

# -----------------------------
# STEP 3: Train/test split (respecting time order!)
# -----------------------------
# CRITICAL for time series: you must NOT randomly shuffle train/test data,
# or the model "cheats" by learning from the future. Split chronologically instead.

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTraining on {len(X_train)} days, testing on {len(X_test)} days")

# -----------------------------
# STEP 4: Train the model
# -----------------------------
model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# STEP 5: Evaluate — how good is it, honestly?
# -----------------------------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error on test set: {mae:.4f} ({mae*100:.2f}%)")

# Compare to a naive baseline: "predict zero change" — if our model isn't
# meaningfully better than this, it has no real predictive edge.
baseline_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))
print(f"Naive baseline (predict 0% change) MAE: {baseline_mae:.4f} ({baseline_mae*100:.2f}%)")

# Directional accuracy: did it at least get up/down right more than 50% of the time?
direction_correct = np.mean(np.sign(predictions) == np.sign(y_test))
print(f"Directional accuracy: {direction_correct*100:.1f}% (50% = coin flip)")

# -----------------------------
# STEP 6: Make a "next month" prediction using the most recent data
# -----------------------------
latest_features = X.iloc[[-1]]
next_month_predicted_return = model.predict(latest_features)[0]
current_price = df["Close"].iloc[-1].iloc[0]
predicted_price = current_price * (1 + next_month_predicted_return)

print(f"\nCurrent price: ${current_price:.2f}")
print(f"Model's predicted return over next ~{HORIZON_DAYS} trading days (~1 month): {next_month_predicted_return*100:.2f}%")
print(f"Implied predicted price: ${predicted_price:.2f}")
print("\n⚠️  Reminder: check the directional accuracy and MAE above against the")
print("    naive baseline before trusting this number for anything real.")
print("    Note: a 1-month horizon leaves much more time for unpredictable")
print("    news/events to occur than a 1-week horizon did — expect this")
print("    model's error to be larger than the weekly version's.")
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


# ==========================================
# STOCK PREDICTION AI - ML MODEL TRAINING
# ==========================================

print("=" * 60)
print("       STOCK PREDICTION AI - ML TRAINING")
print("=" * 60)

# ------------------------------------------
# 1. STOCK SYMBOL
# ------------------------------------------

SYMBOL = "AAPL"

print(f"\nDownloading historical data for {SYMBOL}...")

# ------------------------------------------
# 2. DOWNLOAD HISTORICAL DATA
# ------------------------------------------

data = yf.download(
    SYMBOL,
    period="5y",
    interval="1d",
    auto_adjust=False,
    progress=False
)

if data.empty:
    print("ERROR: No stock data found.")
    exit()

# Handle MultiIndex columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.reset_index()

print(f"Downloaded {len(data)} rows of data.")

# ------------------------------------------
# 3. SELECT REQUIRED COLUMNS
# ------------------------------------------

required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

data = data[required_columns].copy()

# Convert numeric columns
numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

# ------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------

print("\nCreating features...")

# Previous day's closing price
data["Previous_Close"] = data["Close"].shift(1)

# Daily return
data["Daily_Return"] = (
    data["Close"] - data["Previous_Close"]
) / data["Previous_Close"]

# 5-day moving average
data["MA_5"] = (
    data["Close"]
    .rolling(window=5)
    .mean()
)

# 10-day moving average
data["MA_10"] = (
    data["Close"]
    .rolling(window=10)
    .mean()
)

# Daily price range
data["Price_Range"] = (
    data["High"] - data["Low"]
)

# ------------------------------------------
# 5. TARGET
# ------------------------------------------

# Predict NEXT trading day's closing price
data["Target"] = data["Close"].shift(-1)

# Remove missing values
data = data.dropna().reset_index(drop=True)

print(f"Usable training rows: {len(data)}")

# ------------------------------------------
# 6. FEATURES
# ------------------------------------------

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Previous_Close",
    "Daily_Return",
    "MA_5",
    "MA_10",
    "Price_Range"
]

X = data[features]
y = data["Target"]

# ------------------------------------------
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# ------------------------------------------

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining data:", len(X_train))
print("Testing data :", len(X_test))

# ------------------------------------------
# 8. RANDOM FOREST MODEL
# ------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# ------------------------------------------
# 9. MODEL PREDICTION
# ------------------------------------------

predictions = model.predict(X_test)

# ------------------------------------------
# 10. MODEL EVALUATION
# ------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("              MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ------------------------------------------
# 11. SAVE MODEL
# ------------------------------------------

model_path = "stock_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully!")
print(f"File: {model_path}")

# ------------------------------------------
# 12. LATEST STOCK PREDICTION
# ------------------------------------------

latest_features = X.iloc[[-1]]

next_price = model.predict(
    latest_features
)[0]

current_price = float(
    data["Close"].iloc[-1]
)

difference = next_price - current_price

if current_price != 0:
    percentage = (
        difference / current_price
    ) * 100
else:
    percentage = 0


print("\n" + "=" * 60)
print("             NEXT DAY PREDICTION")
print("=" * 60)

print(f"Stock          : {SYMBOL}")
print(f"Current Price  : ${current_price:.2f}")
print(f"Predicted Price: ${next_price:.2f}")
print(f"Expected Change: ${difference:.2f}")
print(f"Expected %     : {percentage:.2f}%")

if percentage > 0.5:
    signal = "BUY"
elif percentage < -0.5:
    signal = "SELL"
else:
    signal = "HOLD"

print(f"AI Signal      : {signal}")

print("=" * 60)
print("\nTraining completed successfully!")
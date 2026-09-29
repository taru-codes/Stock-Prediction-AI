from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "stock_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "stock_model.pkl not found. Keep app.py and stock_model.pkl in the same folder."
    )

model = joblib.load(MODEL_PATH)


# ================= HOME PAGE =================

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/index.html")
@app.route("/index.html/")
def index_page():
    return send_from_directory(BASE_DIR, "index.html")


# ================= ABOUT PAGE =================

@app.route("/about.html")
@app.route("/about.html/")
def about_page():
    return send_from_directory(BASE_DIR, "about.html")


# ================= FEATURES PAGE =================

@app.route("/features.html")
@app.route("/features.html/")
def features_page():
    return send_from_directory(BASE_DIR, "features.html")


# ================= MODELS PAGE =================

@app.route("/models.html")
@app.route("/models.html/")
def models_page():
    return send_from_directory(BASE_DIR, "models.html")


# ================= WORKFLOW PAGE =================

@app.route("/workflow.html")
@app.route("/workflow.html/")
def workflow_page():
    return send_from_directory(BASE_DIR, "workflow.html")


# ================= STOCK PREDICTION API =================

@app.route("/api/predict")
def predict():

    symbol = request.args.get(
        "symbol",
        "AAPL"
    ).upper().strip()

    try:

        print(f"\nFetching stock data for {symbol}...")

        data = yf.download(
            symbol,
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            return jsonify({
                "status": "error",
                "message": f"No stock data found for {symbol}"
            }), 404

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        data = data[required_columns].copy()

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

        data["Previous_Close"] = data["Close"].shift(1)

        data["Daily_Return"] = (
            data["Close"] - data["Previous_Close"]
        ) / data["Previous_Close"]

        data["MA_5"] = (
            data["Close"]
            .rolling(window=5)
            .mean()
        )

        data["MA_10"] = (
            data["Close"]
            .rolling(window=10)
            .mean()
        )

        data["Price_Range"] = (
            data["High"] - data["Low"]
        )

        data = (
            data
            .dropna()
            .reset_index(drop=True)
        )

        if data.empty:
            return jsonify({
                "status": "error",
                "message": "Not enough data for prediction."
            }), 400

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

        latest_features = (
            data[features]
            .iloc[[-1]]
        )

        predicted_price = float(
            model.predict(latest_features)[0]
        )

        current_price = float(
            data["Close"].iloc[-1]
        )

        day_high = float(
            data["High"].iloc[-1]
        )

        day_low = float(
            data["Low"].iloc[-1]
        )

        difference = (
            predicted_price - current_price
        )

        percentage = (
            difference / current_price
        ) * 100 if current_price != 0 else 0

        if percentage > 0.5:
            signal = "BUY"

        elif percentage < -0.5:
            signal = "SELL"

        else:
            signal = "HOLD"

        return jsonify({

            "status": "success",

            "symbol": symbol,

            "current_price": round(
                current_price,
                2
            ),

            "predicted_price": round(
                predicted_price,
                2
            ),

            "change": round(
                difference,
                2
            ),

            "change_percent": round(
                percentage,
                2
            ),

            "high": round(
                day_high,
                2
            ),

            "low": round(
                day_low,
                2
            ),

            "signal": signal

        })

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ================= START SERVER =================

if __name__ == "__main__":

    print("=" * 60)
    print("       STOCK PREDICTION AI - FLASK BACKEND")
    print("=" * 60)

    print("\nServer starting...")

    print("\nWebsite:")
    print("http://127.0.0.1:5000")

    print("\nPrediction API:")
    print("http://127.0.0.1:5000/api/predict?symbol=AAPL")

    print("\nServer is ready!")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
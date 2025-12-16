from flask import Flask, jsonify, request
from pathlib import Path
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression

# prophet is optional
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

app = Flask(__name__)

DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
WINNERS_PATH = Path("data/processed/plots/model_winners.json")

df_global = pd.read_parquet(DATA_PATH)
winners_df = pd.read_json(WINNERS_PATH)


# health check route
@app.route("/", methods=["GET"])
def index():
    return "Job Market Analysis API is running"

# list job titles with enough history
@app.route("/api/titles", methods=["GET"])
def get_titles():
    counts = (
        df_global.groupby("job_title")["month"]
        .nunique()
        .reset_index(name="months")
    )
    valid_titles = (
        counts[counts["months"] >= 8]
        .sort_values("job_title")["job_title"]
        .tolist()
    )
    return jsonify({"titles": valid_titles})

@app.route("/api/history", methods=["GET"])
def get_history():
    title = request.args.get("title", type=str)

    if not title:
        return jsonify({"error": "Missing 'title' parameter"}), 400

    # filter data
    data = df_global[df_global["job_title"].str.lower() == title.lower()].copy()

    if data.empty:
        return jsonify({"error": f"No data found for title: {title}"}), 404

    # aggregate salary per month
    monthly = (
        data.groupby("month", as_index=False)
            .agg(avg_salary=("avg_salary", "mean"))
            .sort_values("month")
    )

    # convert Timestamp → string for JSON
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m-01")

    return jsonify({
        "job_title": title,
        "history": monthly.to_dict(orient="records")
    })


@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    title = request.args.get("title", type=str)
    horizon = request.args.get("horizon", default=6, type=int)

    if not title:
        return jsonify({"error": "Missing 'title' parameter"}), 400

    # get best model for this title
    row = winners_df[winners_df["job_title"].str.lower() == title.lower()]
    if row.empty:
        return jsonify({"error": f"No winner model found for title: {title}"}), 404

    best_model = row.iloc[0]["best_model"]

    # build monthly series
    data = df_global[df_global["job_title"].str.lower() == title.lower()].copy()
    if data.empty:
        return jsonify({"error": f"No data found for title: {title}"}), 404

    monthly = (
        data.groupby("month", as_index=False)
            .agg(avg_salary=("avg_salary", "mean"))
            .sort_values("month")
            .reset_index(drop=True)
    )

    if monthly["month"].nunique() < 8:
        return jsonify({"error": "Insufficient history (< 8 months) for forecasting"}), 400

    # Linear model
    if best_model == "Linear":
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly["avg_salary"].values

        model = LinearRegression().fit(X, y)

        future_idx = np.arange(len(monthly), len(monthly) + horizon).reshape(-1, 1)
        preds = model.predict(future_idx)

        dates = pd.date_range(
            start=monthly["month"].max() + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS"
        )

        forecast = [
            {
                "month": d.strftime("%Y-%m-01"),
                "predicted_salary": float(p)
            }
            for d, p in zip(dates, preds)
        ]

        return jsonify({
            "job_title": title,
            "model": "Linear",
            "forecast": forecast
        })

    # Prophet model
    elif best_model.startswith("Prophet") and Prophet is not None:
        p_df = monthly.rename(columns={"month": "ds", "avg_salary": "y"})[["ds", "y"]]

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.5
        )
        m.fit(p_df)

        future = m.make_future_dataframe(periods=horizon, freq="MS")
        fc = m.predict(future).tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        forecast = [
            {
                "month": r["ds"].strftime("%Y-%m-01"),
                "predicted_salary": float(r["yhat"]),
                "yhat_lower": float(r["yhat_lower"]),
                "yhat_upper": float(r["yhat_upper"])
            }
            for _, r in fc.iterrows()
        ]

        return jsonify({
            "job_title": title,
            "model": "Prophet",
            "forecast": forecast
        })

    else:
        return jsonify({"error": f"Unsupported model or Prophet not available: {best_model}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

from flask import Flask, jsonify, request
from pathlib import Path
import pandas as pd
import numpy as np
import os
from flask_cors import CORS

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
CORS(app)

DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
WINNERS_PATH = Path("data/processed/plots/model_winners.json")
KPI_DIR = Path("data/processed/kpis")

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
    
def _read_kpi_csv(filename: str):
    path = KPI_DIR / filename
    if not path.exists():
        return None, f"Missing KPI file: {path.as_posix()}. Run python backend/etl/kpi_generate.py"
    df_kpi = pd.read_csv(path)
    return df_kpi.to_dict(orient="records"), None


@app.route("/api/kpis", methods=["GET"])
def get_all_kpis():
    # returns everything in one call (easy for dashboard)
    if not KPI_DIR.exists():
        return jsonify({"error": f"Missing KPI folder: {KPI_DIR.as_posix()}"}), 500

    kpi_map = {
        "top_jobs_openings": "top_jobs_openings.csv",
        "top_jobs_salary": "top_jobs_salary.csv",
        "salary_growth_top10": "salary_growth_top10.csv",
        "salary_spikes_top10": "salary_spikes_top10.csv",
        "salary_volatility_top10": "salary_volatility_top10.csv",
        "top_locations_salary": "top_locations_salary.csv",
    }

    out = {}
    for key, fname in kpi_map.items():
        data, err = _read_kpi_csv(fname)
        if err:
            return jsonify({"error": err}), 500
        out[key] = data

    return jsonify(out)


@app.route("/api/kpis/<name>", methods=["GET"])
def get_one_kpi(name):
    # fetch one KPI list by name
    kpi_map = {
        "top_jobs_openings": "top_jobs_openings.csv",
        "top_jobs_salary": "top_jobs_salary.csv",
        "salary_growth_top10": "salary_growth_top10.csv",
        "salary_spikes_top10": "salary_spikes_top10.csv",
        "salary_volatility_top10": "salary_volatility_top10.csv",
        "top_locations_salary": "top_locations_salary.csv",
    }

    if name not in kpi_map:
        return jsonify({
            "error": f"Unknown KPI: {name}",
            "available": sorted(list(kpi_map.keys()))
        }), 404

    data, err = _read_kpi_csv(kpi_map[name])
    if err:
        return jsonify({"error": err}), 500

    return jsonify({"name": name, "data": data})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

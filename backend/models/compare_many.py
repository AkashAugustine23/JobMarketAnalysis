import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
PLOT_DIR = Path("data/processed/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

titles = [
    "Accountable Manager",
    "Assistant Project Manager",
    "Senior Project Manager",
    "Project Manager",
]

def rmse(y_true, y_pred): return float(mean_squared_error(y_true, y_pred) ** 0.5)
def mape(y_true, y_pred): return float(mean_absolute_percentage_error(y_true, y_pred) * 100)

df = pd.read_parquet(DATA_PATH)
records = []

for job_title in titles:
    data = df[df["job_title"].str.lower() == job_title.lower()].copy()
    if data.empty:
        print(f"Skipping (not found): {job_title}")
        continue

    monthly = (
        data.groupby("month", as_index=False)
            .agg(avg_salary=("avg_salary", "mean"))
            .sort_values("month")
            .reset_index(drop=True)
    )
    if monthly["month"].nunique() < 8:
        print(f"Skipping (too short): {job_title} ({monthly['month'].nunique()} months)")
        continue

    monthly["t"] = np.arange(len(monthly))
    X_all = monthly[["t"]].values
    y_all = monthly["avg_salary"].values
    months = monthly["month"].values

    split = int(len(monthly) * 0.8)
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    # Linear
    lin = LinearRegression().fit(X_tr, y_tr)
    y_pred_lin = lin.predict(X_te)
    rmse_lin = rmse(y_te, y_pred_lin)
    mape_lin = mape(y_te, y_pred_lin)

    # Polynomial (deg 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtr_poly = poly.fit_transform(X_tr)
    Xte_poly = poly.transform(X_te)
    poly_model = LinearRegression().fit(Xtr_poly, y_tr)
    y_pred_poly = poly_model.predict(Xte_poly)
    rmse_poly = rmse(y_te, y_pred_poly)
    mape_poly = mape(y_te, y_pred_poly)

    # Prophet
    rmse_prophet = mape_prophet = None
    if Prophet is not None:
        p_df = monthly.rename(columns={"month": "ds", "avg_salary": "y"})[["ds","y"]]
        split_idx = int(len(p_df)*0.8)
        train, test = p_df.iloc[:split_idx].copy(), p_df.iloc[split_idx:].copy()
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.5)
        m.fit(train)
        fut = m.make_future_dataframe(periods=len(test), freq="MS")
        fc_all = m.predict(fut)
        pred_test = fc_all.set_index("ds")[["yhat"]].reindex(test["ds"]).dropna()
        if len(pred_test) == len(test):
            rmse_prophet = rmse(test["y"].values, pred_test["yhat"].values)
            mape_prophet = mape(test["y"].values, pred_test["yhat"].values)

    records.append({
        "job_title": job_title,
        "rmse_linear": rmse_lin, "mape_linear": mape_lin,
        "rmse_poly": rmse_poly, "mape_poly": mape_poly,
        "rmse_prophet": rmse_prophet, "mape_prophet": mape_prophet
    })

# save results table
out_csv = PLOT_DIR / "model_comparison_summary.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"Saved summary: {out_csv}")

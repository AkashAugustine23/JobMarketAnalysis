import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# try prophet import
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

job_title = "Assistant Project Manager"
degree = 2

def rmse(y_true, y_pred): return float(mean_squared_error(y_true, y_pred) ** 0.5)
def mape(y_true, y_pred): return float(mean_absolute_percentage_error(y_true, y_pred) * 100)

# --- Load and aggregate to one row per month (aligns all models) ---
df = pd.read_parquet(DATA_PATH)
data = df[df["job_title"].str.lower() == job_title.lower()].copy()
if data.empty:
    raise ValueError(f"No records found for job title: {job_title}")

# aggregate across locations → one value per month
monthly = (
    data.groupby("month", as_index=False)
        .agg(avg_salary=("avg_salary", "mean"))
        .sort_values("month")
        .reset_index(drop=True)
)
if monthly["month"].nunique() < 8:
    raise ValueError(f"Need at least 8 monthly points, found {monthly['month'].nunique()}.")

# features/targets
monthly["t"] = np.arange(len(monthly))
X_all = monthly[["t"]].values
y_all = monthly["avg_salary"].values
months = monthly["month"].values

# split 80/20
split = int(len(monthly) * 0.8)
X_tr, X_te = X_all[:split], X_all[split:]
y_tr, y_te = y_all[:split], y_all[split:]
months_tr, months_te = months[:split], months[split:]

# --- Linear ---
lin = LinearRegression().fit(X_tr, y_tr)
y_fit_lin = lin.predict(X_all)
y_pred_lin = lin.predict(X_te)
rmse_lin = rmse(y_te, y_pred_lin)
mape_lin = mape(y_te, y_pred_lin)

# --- Polynomial ---
poly = PolynomialFeatures(degree=degree, include_bias=False)
Xtr_poly = poly.fit_transform(X_tr)
Xte_poly = poly.transform(X_te)
poly_model = LinearRegression().fit(Xtr_poly, y_tr)
y_fit_poly = poly_model.predict(poly.transform(X_all))
y_pred_poly = poly_model.predict(Xte_poly)
rmse_poly = rmse(y_te, y_pred_poly)
mape_poly = mape(y_te, y_pred_poly)

# --- Prophet (same monthly series) ---
rmse_prophet = None
mape_prophet = None
y_fit_prophet = None
if Prophet is not None:
    p_df = monthly.rename(columns={"month": "ds", "avg_salary": "y"})[["ds","y"]]
    split_idx = int(len(p_df)*0.8)
    train, test = p_df.iloc[:split_idx].copy(), p_df.iloc[split_idx:].copy()

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(train)

    in_sample = m.predict(pd.DataFrame({"ds": p_df["ds"]}))
    y_fit_prophet = in_sample["yhat"].values

    fut = m.make_future_dataframe(periods=len(test), freq="MS")
    fc_all = m.predict(fut)
    pred_test = (
        fc_all.set_index("ds")[["yhat"]]
        .reindex(test["ds"])
        .dropna()
        .reset_index()
    )
    if len(pred_test) == len(test):
        rmse_prophet = rmse(test["y"].values, pred_test["yhat"].values)
        mape_prophet = mape(test["y"].values, pred_test["yhat"].values)

print("\nModel Comparison (aligned monthly series)")
print("----------------------------------------")
print(f"Job Title: {job_title}")
print(f"Linear         -> RMSE: {rmse_lin:.2f}, MAPE: {mape_lin:.2f}%")
print(f"Polynomial d{degree} -> RMSE: {rmse_poly:.2f}, MAPE: {mape_poly:.2f}%")
if rmse_prophet is not None and mape_prophet is not None:
    print(f"Prophet        -> RMSE: {rmse_prophet:.2f}, MAPE: {mape_prophet:.2f}%")
else:
    print("Prophet        -> skipped (not installed or alignment issue)")

# --- Plot actual + fitted lines (same x-length) ---
plt.figure(figsize=(11,6))
plt.plot(months, y_all, label="Actual", marker="o", color="black")
plt.plot(months, y_fit_lin, label="Linear", linestyle="--", color="blue")
plt.plot(months, y_fit_poly, label=f"Poly (deg {degree})", linestyle="--", color="orange")
if y_fit_prophet is not None:
    plt.plot(months, y_fit_prophet, label="Prophet", linestyle="--", color="green")

plt.title(f"Model Comparison — {job_title}")
plt.xlabel("Month")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

out_path = PLOT_DIR / f"compare_all_{job_title.replace(' ', '_')}.png"
plt.savefig(out_path)
plt.close()
print(f"\n✅ Combined plot saved: {out_path}")

import csv
metrics_path = PLOT_DIR / f"metrics_{job_title.replace(' ', '_')}.csv"
with open(metrics_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","RMSE","MAPE"])
    w.writerow(["Linear", f"{rmse_lin:.2f}", f"{mape_lin:.2f}%"])
    w.writerow([f"Polynomial d{degree}", f"{rmse_poly:.2f}", f"{mape_poly:.2f}%"])
    if rmse_prophet is not None:
        w.writerow(["Prophet", f"{rmse_prophet:.2f}", f"{mape_prophet:.2f}%"])
print(f"Saved metrics: {metrics_path}")

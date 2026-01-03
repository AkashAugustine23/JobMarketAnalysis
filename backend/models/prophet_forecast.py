import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# import prophet (fallback to fbprophet if needed)
try:
    from prophet import Prophet
except Exception:
    from fbprophet import Prophet  # only if your env uses the older package name

# paths
DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
PLOT_DIR = Path("data/processed/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# choose job title
job_title = "Assistant Project Manager"

# load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)

# filter by title
data = df[df["job_title"].str.lower() == job_title.lower()].copy()
if data.empty:
    raise ValueError(f"No records found for job title: {job_title}")

# sort and keep needed columns
data = data.sort_values("month").reset_index(drop=True)

# require enough points
if data["month"].nunique() < 8:
    raise ValueError(f"Need at least 8 monthly points, found {data['month'].nunique()} for {job_title}")

# prepare Prophet dataframe (ds, y)
# Prophet expects a continuous timeline; we aggregate mean salary per month (already monthly)
prophet_df = data[["month", "avg_salary"]].rename(columns={"month": "ds", "avg_salary": "y"})

# optional: handle any duplicate months (take mean)
prophet_df = prophet_df.groupby("ds", as_index=False)["y"].mean()

# split 80/20 for evaluation
split_idx = int(len(prophet_df) * 0.8)
train = prophet_df.iloc[:split_idx].copy()
test = prophet_df.iloc[split_idx:].copy()

# build and fit model
m = Prophet(
    yearly_seasonality=False,  # enable if you have 2+ years of data
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.5
)
m.fit(train)

# in-sample forecast for test horizon
future_test = m.make_future_dataframe(periods=len(test), freq="MS")
forecast_all = m.predict(future_test)
pred_test = forecast_all.tail(len(test))[["ds", "yhat"]].set_index("ds").reindex(test["ds"]).reset_index()

# metrics
def rmse(a, p):
    return float(np.sqrt(np.mean((np.array(a) - np.array(p))**2)))

def mape(a, p):
    a = np.array(a, dtype=float)
    p = np.array(p, dtype=float)
    return float(np.mean(np.abs((a - p) / np.clip(a, 1e-9, None))) * 100)

rmse_val = rmse(test["y"], pred_test["yhat"])
mape_val = mape(test["y"], pred_test["yhat"])

print("\nProphet Evaluation")
print("------------------")
print(f"Job Title: {job_title}")
print(f"RMSE: {rmse_val:.2f}")
print(f"MAPE: {mape_val:.2f}%")

# forecast next 6 months beyond last point
future_h = 6
future = m.make_future_dataframe(periods=future_h, freq="MS")
forecast = m.predict(future)
future_fc = forecast.tail(future_h)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
future_fc = future_fc.rename(columns={"ds": "month", "yhat": "predicted_salary"})

print("\nNext 6 months forecast:")
print(future_fc)

# plot actual vs fitted vs forecast
plt.figure(figsize=(11,6))
plt.plot(prophet_df["ds"], prophet_df["y"], label="Actual", marker="o")
# fitted for all available ds
fitted_merge = forecast_all[["ds", "yhat"]].merge(prophet_df[["ds"]], on="ds", how="right")
plt.plot(fitted_merge["ds"], fitted_merge["yhat"], label="Fitted (Prophet)", linestyle="--")
# forecast band
plt.plot(future_fc["month"], future_fc["predicted_salary"], label="Forecast (Next 6)", marker="x")
plt.fill_between(
    forecast.tail(future_h)["ds"],
    forecast.tail(future_h)["yhat_lower"],
    forecast.tail(future_h)["yhat_upper"],
    alpha=0.2, label="Forecast Interval"
)
plt.title(f"Prophet Forecast — {job_title}")
plt.xlabel("Month")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

out_path = PLOT_DIR / f"prophet_{job_title.replace(' ', '_')}.png"
plt.savefig(out_path)
plt.close()
print(f"\n Plot saved: {out_path}")

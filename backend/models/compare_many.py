import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# try prophet (skip if not installed)
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# paths
DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
PLOT_DIR = Path("data/processed/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# metric helpers
def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred) ** 0.5)

def mape(y_true, y_pred):
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)

# load dataset
df = pd.read_parquet(DATA_PATH)

# automatically select job titles with at least 8 monthly records
titles = (
    df.groupby("job_title")["month"]
      .nunique()
      .reset_index()
      .query("month >= 8")
      ["job_title"]
      .tolist()
)

print("Job titles with >= 8 months of data:")
for t in titles:
    print(" -", t)

records = []

# loop through each job title
for job_title in titles:
    print(f"\nProcessing: {job_title}")

    data = df[df["job_title"].str.lower() == job_title.lower()].copy()
    if data.empty:
        print("  Skipped (not found)")
        continue

    # monthly average
    monthly = (
        data.groupby("month", as_index=False)
            .agg(avg_salary=("avg_salary", "mean"))
            .sort_values("month")
            .reset_index(drop=True)
    )

    if monthly["month"].nunique() < 8:
        print(f"  Skipped (only {monthly['month'].nunique()} months)")
        continue

    # prepare features
    monthly["t"] = np.arange(len(monthly))
    X_all = monthly[["t"]].values
    y_all = monthly["avg_salary"].values

    # split train/test
    split = int(len(monthly) * 0.8)
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    # ---------------- Linear ----------------
    lin = LinearRegression().fit(X_tr, y_tr)
    y_pred_lin = lin.predict(X_te)
    rmse_lin = rmse(y_te, y_pred_lin)
    mape_lin = mape(y_te, y_pred_lin)

    # ---------------- Prophet ----------------
    rmse_prophet = None
    mape_prophet = None

    if Prophet is not None:
        p_df = monthly.rename(columns={"month": "ds", "avg_salary": "y"})[["ds", "y"]]
        split_idx = int(len(p_df) * 0.8)
        train, test = p_df.iloc[:split_idx], p_df.iloc[split_idx:]

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        m.fit(train)

        fut = m.make_future_dataframe(periods=len(test), freq="MS")
        fc_all = m.predict(fut)

        pred_test = (
            fc_all.set_index("ds")[["yhat"]]
                  .reindex(test["ds"])
                  .dropna()
        )

        if len(pred_test) == len(test):
            rmse_prophet = rmse(test["y"].values, pred_test["yhat"].values)
            mape_prophet = mape(test["y"].values, pred_test["yhat"].values)

    # store results
    records.append({
        "job_title": job_title,
        "lin_rmse": rmse_lin, "lin_mape": mape_lin,
        "prophet_rmse": rmse_prophet, "prophet_mape": mape_prophet
    })

# save CSV
out_csv = PLOT_DIR / "model_comparison_summary.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"\nSaved summary: {out_csv}")

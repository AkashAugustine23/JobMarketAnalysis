import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# define paths
DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
PLOT_DIR = Path("data/processed/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# choose job title (change if needed)
job_title = "Assistant Project Manager"

# choose polynomial degree (start with 2 or 3)
degree = 2

# load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)

# filter by job title
data = df[df["job_title"].str.lower() == job_title.lower()].copy()
if data.empty:
    raise ValueError(f"No records found for job title: {job_title}")

# sort by month
data = data.sort_values("month").reset_index(drop=True)

# require enough points
if len(data) < 8:
    raise ValueError(f"Not enough points for polynomial fit (need >= 8, found {len(data)}). Try another title.")

# create time index
data["month_num"] = np.arange(len(data))
X = data[["month_num"]].values
y = data["avg_salary"].values

# train/test split (80/20)
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
months_train = data.loc[:split-1, "month"]
months_test = data.loc[split:, "month"]

# polynomial transform
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_tr = poly.fit_transform(X_train)
X_te = poly.transform(X_test)

# fit model
model = LinearRegression()
model.fit(X_tr, y_train)

# predictions on test
y_pred = model.predict(X_te)

# metrics (RMSE as sqrt(MSE for sklearn compatibility)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("\nPolynomial Regression Results")
print("----------------------------")
print(f"Job Title: {job_title}")
print(f"Degree: {degree}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# predict next 6 months
future_steps = 6
last_idx = data["month_num"].iloc[-1]
future_idx = np.arange(last_idx + 1, last_idx + 1 + future_steps).reshape(-1, 1)
future_X = poly.transform(future_idx)
future_pred = model.predict(future_X)
future_dates = pd.date_range(start=data["month"].max() + pd.offsets.MonthBegin(1), periods=future_steps, freq="MS")

forecast_df = pd.DataFrame({
    "month": future_dates,
    "predicted_salary": future_pred
})

print("\nNext 6 months forecast:")
print(forecast_df)

# plot actual, fitted (train+test), and forecast
plt.figure(figsize=(11,6))
# actual
plt.plot(data["month"], y, label="Actual", marker="o")
# fitted for whole series to show curve
X_all = poly.transform(X)
y_all_pred = model.predict(X_all)
plt.plot(data["month"], y_all_pred, label="Fitted (Polynomial)", linestyle="--")
# forecast
plt.plot(forecast_df["month"], forecast_df["predicted_salary"], label="Forecast (Next 6)", marker="x")
plt.title(f"Polynomial Regression (deg {degree}) — {job_title}")
plt.xlabel("Month")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

out_path = PLOT_DIR / f"poly_regression_deg{degree}_{job_title.replace(' ', '_')}.png"
plt.savefig(out_path)
plt.close()
print(f"\n✅ Plot saved: {out_path}")

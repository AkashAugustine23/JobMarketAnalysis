import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta

# define file paths
DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
MODEL_OUTPUT = Path("data/processed/plots")
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

# load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)

# choose a job title to model (is changeable)
job_title = "Assistant Project Manager"

# filter data for that job title
data = df[df["job_title"].str.lower() == job_title.lower()].copy()
if data.empty:
    raise ValueError(f"No records found for job title: {job_title}")

# sort by month
data = data.sort_values("month")

# convert month to numeric index for regression
data["month_num"] = np.arange(len(data))

# define features and target
X = data[["month_num"]]
y = data["avg_salary"]

# create and train model
model = LinearRegression()
model.fit(X, y)

# predict next 6 months
future_steps = 6
future_months = np.arange(len(data), len(data) + future_steps)
future_dates = pd.date_range(start=data["month"].max() + pd.offsets.MonthBegin(1), periods=future_steps, freq="MS")

future_preds = model.predict(future_months.reshape(-1, 1))

# combine results
forecast_df = pd.DataFrame({
    "month": future_dates,
    "predicted_salary": future_preds
})

# plot results
plt.figure(figsize=(10,6))
plt.plot(data["month"], data["avg_salary"], label="Actual", marker="o")
plt.plot(forecast_df["month"], forecast_df["predicted_salary"], label="Predicted (Next 6 Months)", linestyle="--", marker="x")
plt.title(f"Salary Trend Prediction: {job_title}")
plt.xlabel("Month")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# save plot
output_file = MODEL_OUTPUT / f"linear_regression_{job_title.replace(' ', '_')}.png"
plt.savefig(output_file)
plt.close()

print(f"✅ Model complete. Saved plot: {output_file}")
print("\nPredicted salaries for next 6 months:")
print(forecast_df)

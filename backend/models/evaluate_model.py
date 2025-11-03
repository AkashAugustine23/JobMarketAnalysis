import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# define file path
DATA_PATH = Path("data/processed/monthly_aggregates.parquet")

# choose same job title as your model
job_title = "Quality Assurance Analyst"

# load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)

# filter data for that job title
data = df[df["job_title"].str.lower() == job_title.lower()].copy()
if data.empty:
    raise ValueError(f"No records found for job title: {job_title}")

# sort by month and reset index
data = data.sort_values("month").reset_index(drop=True)

# convert month to numeric for regression
data["month_num"] = np.arange(len(data))

# define features and target
X = data[["month_num"]]
y = data["avg_salary"]

# split into train/test (80/20)
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# calculate metrics
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# print results
print("\nModel Evaluation Results")
print("------------------------")
print(f"Job Title: {job_title}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# compare predicted vs actual values
results = pd.DataFrame({
    "Month": data.loc[split:, "month"].values,
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\nPredicted vs Actual Salaries:")
print(results)

# save results
output_path = Path("data/processed/plots") / f"evaluation_{job_title.replace(' ', '_')}.csv"
results.to_csv(output_path, index=False)
print(f"\n✅ Evaluation complete. Results saved to: {output_path}")

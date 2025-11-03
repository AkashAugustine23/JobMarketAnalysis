import pandas as pd
from pathlib import Path

# load processed data
df = pd.read_parquet(Path("data/processed/monthly_aggregates.parquet"))

# ensure sorted months
df = df.sort_values(["job_title", "month"])

# months of history per title
hist = (
    df.groupby("job_title")
      .agg(months=("month", "nunique"),
           total_posts=("job_count", "sum"),
           first=("month", "min"),
           last=("month", "max"),
           avg_salary=("avg_salary", "mean"))
      .reset_index()
      .sort_values(["months","total_posts"], ascending=[False, False])
)

# show solid candidates (>= 12 months preferred; fall back to >= 8)
print("\nTitles with >= 12 months of data (best for forecasting):")
print(hist[hist["months"] >= 12].head(30).to_string(index=False))

print("\nIf the above is empty/small, titles with >= 8 months of data:")
print(hist[(hist["months"] >= 8) & (hist["months"] < 12)].head(30).to_string(index=False))

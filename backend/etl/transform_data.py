import pandas as pd
from pathlib import Path

# define file paths
RAW_PATH = Path("data/raw/job_data_final.xlsx")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# load dataset
print(f"Reading dataset: {RAW_PATH}")
df = pd.read_excel(RAW_PATH)

# standardize column names
df.columns = [c.strip().lower() for c in df.columns]
df = df.rename(columns={
    "business title": "job_title",
    "salary per annum": "salary_annual",
    "posting date": "posting_date",
    "work location": "work_location"
})

print("\nColumns standardized:")
print(df.columns.tolist())

# basic cleaning
df = df.dropna(subset=["job_title", "salary_annual", "posting_date"])
df["salary_annual"] = pd.to_numeric(df["salary_annual"], errors="coerce")
df = df[df["salary_annual"] > 0]

df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
df = df.dropna(subset=["posting_date"])
df["month"] = df["posting_date"].dt.to_period("M").dt.to_timestamp()

# aggregate data (monthly)
grouped = (
    df.groupby(["month", "job_title", "work_location"])
    .agg(
        job_count=("job_title", "count"),
        avg_salary=("salary_annual", "mean")
    )
    .reset_index()
)

print("\nAggregation complete.")
print(f"Shape: {grouped.shape}")
print("Columns:", grouped.columns.tolist())

# save output
output_path = PROCESSED_DIR / "monthly_aggregates.parquet"
grouped.to_parquet(output_path, index=False)

print(f"\n Processed data saved to: {output_path}")
print("ETL process completed successfully.")

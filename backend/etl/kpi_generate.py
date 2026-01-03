import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/monthly_aggregates.parquet")
OUT_DIR = Path("data/processed/kpis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)

# --- KPI 1: Top 10 jobs by total openings ---
top_jobs_openings = (
    df.groupby("job_title", as_index=False)["job_count"]
      .sum()
      .sort_values("job_count", ascending=False)
      .head(10)
)

# --- KPI 2: Top 10 jobs by highest average salary ---
top_jobs_salary = (
    df.groupby("job_title", as_index=False)["avg_salary"]
      .mean()
      .sort_values("avg_salary", ascending=False)
      .head(10)
)

# --- KPI 3: Salary growth leaders (first vs last month %) ---
tmp = df.sort_values(["job_title", "month"]).copy()
first_last = (
    tmp.groupby("job_title")
       .agg(first_salary=("avg_salary", "first"),
            last_salary=("avg_salary", "last"))
       .reset_index()
)
first_last["growth_pct"] = (first_last["last_salary"] - first_last["first_salary"]) / first_last["first_salary"] * 100
salary_growth = (
    first_last.replace([float("inf"), float("-inf")], pd.NA)
              .dropna(subset=["growth_pct"])
              .sort_values("growth_pct", ascending=False)
              .head(10)
)

# --- KPI 4: Salary spike events (largest MoM % change) ---
tmp["salary_pct_change"] = tmp.groupby("job_title")["avg_salary"].pct_change() * 100
salary_spikes = (
    tmp.dropna(subset=["salary_pct_change"])
       .sort_values("salary_pct_change", ascending=False)
       .head(10)[["job_title", "month", "avg_salary", "salary_pct_change"]]
)

# --- KPI 5: Most volatile salaries (std dev) ---
salary_volatility = (
    df.groupby("job_title", as_index=False)["avg_salary"]
      .std()
      .rename(columns={"avg_salary": "salary_std"})
      .sort_values("salary_std", ascending=False)
      .head(10)
)

# --- KPI 6: Top locations by salary ---
top_locations_salary = (
    df.groupby("work_location", as_index=False)["avg_salary"]
      .mean()
      .sort_values("avg_salary", ascending=False)
      .head(10)
)

# Save all as CSV (easy for dashboard + report later)
top_jobs_openings.to_csv(OUT_DIR / "top_jobs_openings.csv", index=False)
top_jobs_salary.to_csv(OUT_DIR / "top_jobs_salary.csv", index=False)
salary_growth.to_csv(OUT_DIR / "salary_growth_top10.csv", index=False)
salary_spikes.to_csv(OUT_DIR / "salary_spikes_top10.csv", index=False)
salary_volatility.to_csv(OUT_DIR / "salary_volatility_top10.csv", index=False)
top_locations_salary.to_csv(OUT_DIR / "top_locations_salary.csv", index=False)

print("KPI files saved to:", OUT_DIR)
print("Files created:")
for p in OUT_DIR.glob("*.csv"):
    print(" -", p.as_posix())

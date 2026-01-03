import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# define file paths
PROCESSED_PATH = Path("data/processed/monthly_aggregates.parquet")
PLOT_DIR = Path("data/processed/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# load processed dataset
print(f"Loading processed data from: {PROCESSED_PATH}")
df = pd.read_parquet(PROCESSED_PATH)

# top 10 job titles by total postings
top_jobs = (
    df.groupby("job_title")["job_count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
top_jobs.plot(kind='bar', color='skyblue')
plt.title("Top 10 Job Titles by Job Count")
plt.xlabel("Job Title")
plt.ylabel("Total Postings")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOT_DIR / "top_job_titles.png")
plt.close()
print("Saved: top_job_titles.png")

# top 10 locations by job count
top_locations = (
    df.groupby("work_location")["job_count"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
top_locations.plot(kind='bar', color='lightgreen')
plt.title("Top 10 Locations by Job Count")
plt.xlabel("Work Location")
plt.ylabel("Total Postings")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOT_DIR / "top_locations.png")
plt.close()
print("Saved: top_locations.png")

# top 10 job titles by average salary
avg_salary = (
    df.groupby("job_title")["avg_salary"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
avg_salary.plot(kind='bar', color='salmon')
plt.title("Top 10 Job Titles by Average Salary")
plt.xlabel("Job Title")
plt.ylabel("Average Salary (Annual)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOT_DIR / "top_salary_titles.png")
plt.close()
print("Saved: top_salary_titles.png")

# salary trend over time for a popular job title
selected_title = top_jobs.index[0]  # most common job
trend = df[df["job_title"] == selected_title].groupby("month")["avg_salary"].mean()

plt.figure(figsize=(10,6))
plt.plot(trend.index, trend.values, marker='o', color='orange')
plt.title(f"Salary Trend Over Time: {selected_title}")
plt.xlabel("Month")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"salary_trend_{selected_title.replace(' ', '_')}.png")
plt.close()
print(f"Saved: salary_trend_{selected_title.replace(' ', '_')}.png")

print("\n Visualization complete. All plots saved in data/processed/plots/")

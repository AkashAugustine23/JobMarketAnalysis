import pandas as pd
from pathlib import Path

# Path to your cleaned dataset
DATA_PATH = Path("data/raw/job_data_final.xlsx")

# Load the Excel file
print(f"Reading dataset: {DATA_PATH}")
df = pd.read_excel(DATA_PATH)

# Basic info
print("\n--- Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(list(df.columns))

print("\n--- Sample rows ---")
print(df.head(5))

# Confirm that your key columns exist
important_cols = ["salary per annum", "business title", "posting date", "work location"]
print("\n--- Checking for key columns ---")
for col in important_cols:
    if any(col.lower() == c.lower() for c in df.columns):
        print(f"Found: {col}")
    else:
        print(f"Missing: {col}")

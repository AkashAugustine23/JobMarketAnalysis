import pandas as pd
from pathlib import Path
import json

# paths
SUMMARY_PATH = Path("data/processed/plots/model_comparison_summary.csv")
OUT_CSV = Path("data/processed/plots/model_winners.csv")
OUT_JSON = Path("data/processed/plots/model_winners.json")

# load summary from compare_many.py
df = pd.read_csv(SUMMARY_PATH)

rows = []

for _, r in df.iterrows():
    job_title = r["job_title"]

    # collect candidate models with their MAPE and RMSE
    candidates = []

    # linear
    if pd.notna(r.get("lin_mape")):
        candidates.append(("Linear", r["lin_mape"], r["lin_rmse"]))

    # polynomial (deg 2)
    if pd.notna(r.get("poly_mape")):
        candidates.append(("Polynomial d2", r["poly_mape"], r["poly_rmse"]))

    # prophet
    if pd.notna(r.get("prophet_mape")):
        candidates.append(("Prophet", r["prophet_mape"], r["prophet_rmse"]))

    # pick best model
    best_model = "N/A"
    best_mape = None
    best_rmse = None

    # filter out rows with NaN MAPE
    valid_by_mape = [c for c in candidates if pd.notna(c[1])]

    if valid_by_mape:
        # choose the one with lowest MAPE
        best = min(valid_by_mape, key=lambda x: x[1])
        best_model, best_mape, best_rmse = best
    else:
        # if all MAPE are NaN, fall back to lowest RMSE (if available)
        valid_by_rmse = [c for c in candidates if pd.notna(c[2])]
        if valid_by_rmse:
            best = min(valid_by_rmse, key=lambda x: x[2])
            best_model, best_mape, best_rmse = best

    rows.append({
        "job_title": job_title,
        "best_model": best_model,
        "best_mape": best_mape,
        "best_rmse": best_rmse
    })

# create dataframe of winners
winners = pd.DataFrame(rows).sort_values(
    ["best_model", "best_mape"], na_position="last"
)

# save CSV
winners.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)

# save JSON
with open(OUT_JSON, "w") as f:
    json.dump(rows, f, indent=2, default=float)
print("Saved:", OUT_JSON)

print("\nWinners preview:")
print(winners.head(10).to_string(index=False))

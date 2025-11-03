import pandas as pd
from pathlib import Path

summary_path = Path("data/processed/plots/model_comparison_summary.csv")
out_csv = Path("data/processed/plots/model_winners.csv")
out_json = Path("data/processed/plots/model_winners.json")

df = pd.read_csv(summary_path)

# melt to long for easy argmin by MAPE (fallback to RMSE if MAPE is NaN)
rows = []
for _, r in df.iterrows():
    cand = []
    if pd.notna(r.get("mape_linear")):
        cand.append(("Linear", r["mape_linear"], r["rmse_linear"]))
    if pd.notna(r.get("mape_poly")):
        cand.append(("Polynomial d2", r["mape_poly"], r["rmse_poly"]))
    if pd.notna(r.get("mape_prophet")):
        cand.append(("Prophet", r["mape_prophet"], r["rmse_prophet"]))
    # choose by lowest MAPE; if tied/NaN, use RMSE
    cand = [c for c in cand if pd.notna(c[1])]
    if not cand:
        cand = [c for c in cand if pd.notna(c[2])]
        best = min(cand, key=lambda x: x[2]) if cand else ("N/A", None, None)
    else:
        best = min(cand, key=lambda x: x[1])
    rows.append({
        "job_title": r["job_title"],
        "best_model": best[0],
        "best_mape": best[1],
        "best_rmse": best[2]
    })

winners = pd.DataFrame(rows).sort_values(["best_model","best_mape"], na_position="last")
winners.to_csv(out_csv, index=False)
winners.to_json(out_json, orient="records", indent=2)

print("Saved:", out_csv)
print("Saved:", out_json)
print("\nWinners preview:")
print(winners.head(10).to_string(index=False))

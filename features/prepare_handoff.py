
from pathlib import Path
import pandas as pd

results = Path("results")

source = results / "candidate_features.csv"
coverage_source = results / "candidate_coverage.csv"

df = pd.read_csv(source, dtype={"trip_id": str})
coverage = pd.read_csv(coverage_source, dtype={"trip_id": str})

# Restore leading zeros in trip IDs.
df["trip_id"] = df["trip_id"].str.zfill(8)
coverage["trip_id"] = coverage["trip_id"].str.zfill(8)

# Final handoff schema: metadata, features, label.
columns = [
    "trip_id",
    "observation_index",
    "arc_id",
    "distance_m",
    "heading_difference",
    "speed_mps",
    "time_gap_s",
    "segment_length_m",
    "candidate_rank",
    "projection_fraction",
    "heading_reliable",
    "label"
]

missing = [column for column in columns if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

handoff = df[columns].copy()

# Validate labels.
if not handoff["label"].isin([0, 1]).all():
    raise ValueError("Invalid or missing labels detected.")

handoff["label"] = handoff["label"].astype(int)

# Save clean handoff files.
handoff.to_csv(
    results / "candidate_features_handoff.csv",
    index=False
)

coverage.to_csv(
    results / "candidate_coverage_handoff.csv",
    index=False
)

print("\n========== HANDOFF SUMMARY ==========")
print("Rows:", len(handoff))
print("Columns:", len(handoff.columns))
print("Unique trips:", handoff["trip_id"].nunique())
print("Positive labels:", (handoff["label"] == 1).sum())
print("Negative labels:", (handoff["label"] == 0).sum())
print("Trip ID example:", handoff.iloc[0]["trip_id"])
print("\nFinal columns:")
print(handoff.columns.tolist())

print("\nFiles ready:")
print("results/candidate_features_handoff.csv")
print("results/candidate_coverage_handoff.csv")

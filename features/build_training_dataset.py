
from pathlib import Path
import pandas as pd

from features.candidate_generation import process_trip as generate_candidates
from features.feature_engineering import process_trip as engineer_features
from features.label_generation import process_trip as generate_labels


DATASET_DIR = Path(r"C:\Users\dhanu\Desktop\Student folder\train")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

trip_dirs = sorted(
    path for path in DATASET_DIR.iterdir()
    if path.is_dir()
)

all_training = []
all_reports = []
failed_trips = []

print(f"Found {len(trip_dirs)} training trips.")

for index, trip_dir in enumerate(trip_dirs, start=1):
    trip_id = trip_dir.name

    print(f"\n[{index}/{len(trip_dirs)}] Processing {trip_id}")

    candidates_file = RESULTS_DIR / f"candidates_{trip_id}.csv"
    features_file = RESULTS_DIR / f"features_{trip_id}.csv"
    training_file = RESULTS_DIR / f"training_{trip_id}.csv"
    report_file = RESULTS_DIR / f"coverage_{trip_id}.csv"

    try:
        # Step 1: Generate road candidates
        generate_candidates(
            str(trip_dir),
            str(candidates_file),
            5
        )

        # Step 2: Engineer candidate features
        engineer_features(
            str(trip_dir),
            str(candidates_file),
            str(features_file)
        )

        # Step 3: Align reference route and generate labels
        generate_labels(
            str(trip_dir),
            str(features_file),
            str(training_file),
            str(report_file)
        )

        training = pd.read_csv(training_file)
        report = pd.read_csv(report_file)

        if training.empty:
            raise ValueError("Training CSV is empty.")

        all_training.append(training)
        all_reports.append(report)

        print(f"SUCCESS: {trip_id}")

    except Exception as error:
        failed_trips.append({
            "trip_id": trip_id,
            "error": str(error)
        })
        print(f"FAILED: {trip_id} - {error}")


if not all_training:
    raise RuntimeError("No trips were successfully processed.")

# Combine successful trips
combined_training = pd.concat(
    all_training,
    ignore_index=True
)

combined_reports = pd.concat(
    all_reports,
    ignore_index=True
)

# Save final datasets
combined_training.to_csv(
    RESULTS_DIR / "candidate_features.csv",
    index=False
)

combined_reports.to_csv(
    RESULTS_DIR / "candidate_coverage.csv",
    index=False
)

# Save failures separately
if failed_trips:
    pd.DataFrame(failed_trips).to_csv(
        RESULTS_DIR / "failed_trips.csv",
        index=False
    )

print("\n========== FINAL DATASET SUMMARY ==========")
print(f"Total trips found: {len(trip_dirs)}")
print(f"Successful trips: {len(all_training)}")
print(f"Failed trips: {len(failed_trips)}")
print(f"Total training rows: {len(combined_training)}")
print(f"Positive labels: {(combined_training['label'] == 1).sum()}")
print(f"Negative labels: {(combined_training['label'] == 0).sum()}")

print("\nGenerated files:")
print("results/candidate_features.csv")
print("results/candidate_coverage.csv")

if failed_trips:
    print("results/failed_trips.csv")

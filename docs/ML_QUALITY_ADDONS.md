# PathSnap Team 14 — additional ML quality checks

These scripts add **inference handoff verification** and **trip-wise robustness evidence**. They reuse the existing saved model, real candidate feature CSV, model factory and fold-safe imputation. Do not retrain all five models or change the chosen deployment model.

## 1. Handoff inference smoke test (fast)

From the repository root:

```bash
source .venv/bin/activate
python scripts/verify_ml_inference.py \
  --model artifacts/handoff/deploy_model.joblib \
  --csv data/handoff/candidate_features_handoff.csv
```

Outputs under `results/ml_inference_audit/`:
- `candidate_scores_<trip>.csv`: `trip_id`, `observation_index`, `arc_id`, `candidate_score`, `emission_log_score`
- `top1_candidates_<trip>.csv`: point-wise classifier-only selections for comparison against Viterbi
- `inference_audit_<trip>.json`: input feature order, missingness, counts and probability sanity checks

**Important:** Scores are class-1 probabilities for each candidate **independently**, so they need not sum to 1 within a GPS observation. Use HMM transition information to decode a full route. This is a smoke test on a training trip, NOT a test-set performance metric. For unlabeled student-testing candidates, provide a separate unlabeled CSV and use `--csv` to point to it.

The script does not send a training `label` column to the estimator or output.

## 2. Optional: out-of-fold per-trip robustness (fits Random Forest in three folds)

```bash
python -u scripts/evaluate_trip_robustness.py \
  --csv data/handoff/candidate_features_handoff.csv \
  --max-obs-per-trip 0 \
  --folds 3 \
  --model random_forest \
  2>&1 | tee trip_robustness.log
```

Outputs under `results/trip_robustness/`:
- `per_trip_oof_metrics.csv`
- `summary.json`
- `trip_top1_distribution.png`

These reflect **held-out folds by trip** from the 80 training trips, but targets are **geometry-derived proxy candidate labels**, not exact ground-truth arc IDs. Do not call them route reconstruction accuracy. The extra run need not change the already-selected model.

## 3. Training dataset row-count comparison

The handoff includes **135,502 distinct GPS observations × 5 nearby road candidates = 677,510 candidate rows** across **80 training trips**. Some teams generate 10 or more candidates per observation, so a dataset exceeding a million rows does not inherently mean more GPS observations or better ground truth. Increasing candidate count should be justified by **reference-route candidate recall / coverage**, extra computation and downstream route-quality validation, not by row count alone.

## 4. Commit only the new code and small generated evidence

```bash
git add scripts/verify_ml_inference.py scripts/evaluate_trip_robustness.py docs/ML_QUALITY_ADDONS.md
git add results/ml_inference_audit/ results/trip_robustness/
git commit -m "Add ML probability handoff checks and trip-wise robustness audit"
git push origin HEAD
```

**Never** add the 225 MB `deploy_model.joblib`, source GPS dataset, or handoff ZIP to ordinary Git without team agreement. Before opening a PR, confirm your branch and check `git status --short`.

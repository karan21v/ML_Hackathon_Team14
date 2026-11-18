# PathSnap Team 14 - Final evaluation readiness check

This package is an **evidence tracker**, not a decoder, not a replacement for Dhanush's code, and not proof of end-to-end functionality. It does **not** train models or fabricate route predictions.

## Install safely

Extract into the root of `ML_Hackathon_Team14`. It **adds**:

- `scripts/final_eval_readiness.py`
- `tests/test_final_eval_readiness.py`
- `FINAL_EVALUATION_NEXT_STEPS.md`

Do not overwrite or delete existing files. Your locally generated `results/route_comparison_00000001.html`, JSON, and untracked `scripts/validate_candidate_handoff.py` can remain untouched.

## Windows PowerShell, from repository root

```powershell
# Be sure you are on your feature branch, not main, before committing:
git branch --show-current

# Test everything:
.\.venv\Scripts\python.exe -m pytest -q

# Inspect available evidence; missing integration steps remain PENDING:
.\.venv\Scripts\python.exe -m scripts.final_eval_readiness --root . --trip 00000001

# Open generated summary in VS Code:
code .\results\final_eval_readiness_00000001.json
```

The report includes `PASS`, `PENDING` and `FAIL`. `PASS` means an *artifact consistency check* succeeded. It does not mean the entire system is implemented. A recorded ML scoring audit does not prove a decoder is available. Missing model files on this laptop are marked `PENDING`, not automatically `FAIL`.

## After teammates provide real outputs

Dhanush: genuine **one-arc-per-observation** decoder CSV, e.g. `results/predicted_00000001.csv` with `trip_id,observation_index,arc_id` (if he provides a full route sequence instead, agree on a different integration contract first).

```powershell
.\.venv\Scripts\python.exe -m data_processing.validate_predictions --zip "data/Student folder.zip" --partition train --trip 00000001 --predictions "results/predicted_00000001.csv" --out results
.\.venv\Scripts\python.exe -m data_processing.compare_routes --zip "data/Student folder.zip" --partition train --trip 00000001 --predictions "results/predicted_00000001.csv" --out results
Invoke-Item .\results\route_comparison_00000001.html
```

Karan: trusted `deploy_model.joblib` (or `best_model.joblib`), the **same feature order** used in training, and an inference run that exports **candidate probabilities**, not merely top-1 winners. The HMM must receive all candidates and their scores to decode with transition constraints.

Rohit: exact FastAPI host/health/prediction endpoints and input/output contract. When the server is actually running, use the **real** health URL:

```powershell
.\.venv\Scripts\python.exe -m scripts.final_eval_readiness --root . --trip 00000001 --api-health-url http://127.0.0.1:8000/health
```

The URL above is merely an example: **do not assume Rohit's app uses `/health`**. On Laptop 2, replace 127.0.0.1 with Laptop 1's actual reachable LAN IP and configured port when testing cross-device connectivity. Verify an actual trip response and take genuine screenshot evidence.

## Final deliverables (official hackathon guide)

1. One **public GitHub repository** containing code for all five models, optimisation, decoder, FastAPI, UI, saved model bundle, dependency setup and Folium plotting.
2. One **consolidated team PDF** (not separate reports) including dataset EDA, five-model comparison, per-model discussion and graphs, Bayesian search, classifier-vs-HMM results, Folium maps, API and two-laptop UI screenshots. Report only observed results and use `N/A - not measured` for gaps.

See `PathSnap_Team14_Final_Report_DRAFT.pdf` separately. It is a **provisional report**, not an adequate final submission until the HMM and API sections are completed, all evidence checked, and the team reviews it for originality/accuracy and AI-use disclosure.

## Suggested clean Git workflow

If the prediction-validation branch has not been committed yet, save those files on its branch before switching. Otherwise, after syncing `main`, use a separate branch for this readiness checker, and commit only the three files in this package. Don't commit the raw dataset ZIP, generated `results/*.html`, or sensitive config.

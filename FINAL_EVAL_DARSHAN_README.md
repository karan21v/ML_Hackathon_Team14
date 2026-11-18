# Team 14 — Darshan final-evaluation work: route comparison and checks

This package adds two new files to the existing PathSnap repository. It **does not** replace the current dataset loader, your EDA code, Dhanush's feature code, or Karan's trained model code.

## Add to your Windows VS Code project

1. Confirm you're working on branch `feature/darshan-final-evaluation`.
2. Extract the package ZIP **into the root** of `ML_Hackathon_Team14` so the new files land at:
   - `data_processing/compare_routes.py`
   - `tests/test_compare_routes.py`
3. Keep the already tracked `data_processing/load_data.py` as-is.
4. Run tests from the repo root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_compare_routes.py
```

## First run: reference map, before your team has predictions

```powershell
.\.venv\Scripts\python.exe -m data_processing.compare_routes --zip "data/Student folder.zip" --partition train --trip 00000001 --out results
Invoke-Item .\results\route_comparison_00000001.html
```

Expected: blue observed GPS points, green **supplied reference** route, toggleable sampled grey roads. The map explicitly states **no predictions supplied**. This is only a map-readiness test, **not** ML evaluation.

Outputs:
- `results/route_comparison_00000001.html` — interactive map
- `results/route_comparison_00000001.json` — checks, counts and provenance

For a map without internet tiles, `--tiles offline` is the default. `--tiles cartodb` is optional when external map tiles work.

## After Dhanush supplies *real decoded* arc IDs

Ask him for a CSV with exactly one predicted row per GPS observation (or a subset if some observations cannot be decoded):

```
trip_id,observation_index,arc_id
00000001,0,<actual_predicted_arc_id>
00000001,1,<actual_predicted_arc_id>
...
```

**Do not literally use the placeholders above as predictions.** `arc_id` must be the row index into that trip's `.arcs` file. Acceptable CSV format is also just `observation_index,arc_id`, or only `arc_id` **in observation order**. Optional `obs_idx` is understood in place of `observation_index`.

Place the actual decoder CSV at `results/predicted_00000001.csv` and run:

```powershell
.\.venv\Scripts\python.exe -m data_processing.compare_routes --zip "data/Student folder.zip" --partition train --trip 00000001 --predictions "results/predicted_00000001.csv" --out results
Invoke-Item .\results\route_comparison_00000001.html
```

The map will add a red layer for predicted arcs. It does not fabricate road connections or infer missing intermediate arcs. It rejects invalid road IDs and duplicate predictions for the same GPS observation.

**Do not** pass a raw five-candidate-per-GPS feature CSV to `--predictions`: this expects the final decoded output, not candidate scores.

## Metrics — avoid misleading the evaluator

- The `.route` file is an ordered **arc sequence**, not one ground-truth arc for each GPS observation. Never compute observation accuracy by comparing the two files row by row.
- The diagnostic for immediately touching adjacent predicted arcs **is not a full continuity score**: sparse GPS may skip intermediate road segments.
- Arc-set overlap in the JSON is **descriptive** and is not point-level accuracy.
- No evaluator-reserved data should be used for development, labels, validation or visuals.
- For valid observation-level accuracy, use independently verified reference-to-observation alignment, document excluded samples, and never leak reference route information into inference.

## Commit your own new code only

```powershell
git status
git add data_processing/compare_routes.py tests/test_compare_routes.py
git commit -m "Add final evaluation route comparison and tests"
git push -u origin feature/darshan-final-evaluation
```

Open a Pull Request from your branch to `main` and ask for review. Your previously untracked `scripts/validate_candidate_handoff.py` will remain uncommitted unless you separately decide to add it after refining its strict missing-value checks.

The code in this package was tested against **synthetic** tracks and predictions (5/5 tests passing); your team should test it against an actual permitted trip before claiming end-to-end correctness. Review and understand the code, and disclose AI assistance as required by the hackathon.

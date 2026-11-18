# Karan: ML owner — fastest safe workflow

## Your exact objective

Create reproducible candidate scoring for GPS observation–road arc pairs. Compare five models (Logistic Regression, Decision Tree, Random Forest, MLP, HistGradientBoosting) on the same trip-grouped folds, select by the predefined validation metric, save each fitted model and tune the selected family with Optuna TPE. Never train on evaluator-reserved trips.

## Steps (from project root)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Put the extracted train/ and student_test/ directories under data/
PYTHONPATH=src pytest -q
```

**Quick full-five-model run** (tests that integration works; NOT comprehensive evaluation):

```bash
python scripts/train_models.py --data-root data --max-train-trips 6 --max-test-trips 2 --max-obs-per-trip 100 --folds 2
```

**Stronger run once everything is stable:**

```bash
python scripts/train_models.py --data-root data --max-train-trips 0 --max-test-trips 10 --max-obs-per-trip 350 --folds 3
```

This uses all 80 training trips with up to 350 spread-out observations per trip. The selection is based only on train-trip grouped CV; student-test is a permitted development-stage check. An optional *more complete* experiment uses `--max-obs-per-trip 0` if computing time permits, but this is genuinely different data and correctly needs retraining.

**Do not waste time repeating the same run:**

```bash
# Exactly the same command as your previous run:
python scripts/train_models.py --data-root data --max-train-trips 0 --max-test-trips 10 --max-obs-per-trip 350 --folds 3
# You should see [CACHE] and [RESUME] messages.
```

**Tune the selected model** (set initial total trials to 8; increase only if time):

```bash
python scripts/tune_model.py --data-root data --trials 8
# Same command again: 0 extra completed trials, reuse tuned model
```

Report outputs: `results/model_comparison.csv`, `results/candidate_coverage.json`, `results/selected_model.json`, `results/optuna_trials.csv`. Models: `artifacts/best_model.joblib`, `artifacts/tuned_model.joblib`, individual baseline models. The `artifacts` folder is gitignored to prevent accidental large pushes; publication depends on hackathon instructions and team repository size limits.

## Most important handoff from Dhanush

Each candidate row must have feature values in the exact `FEATURE_NAMES` order in `src/pathsnap/features.py`, plus `trip_id`, `obs_idx`, `arc_id`, and `label` as non-feature metadata. The `label` is an approximate GPS-to-reference-route alignment; the raw `.route` provides arc order, not pointwise truth. Do NOT use `.route`, `arc_id` or identifiers as inference model inputs. Verify this before trusting metrics.

## Repetition vs legitimate multiple fits

A 3-fold CV experiment trains 3 independent models per algorithm plus 1 final model, so five algorithms imply **20 actual fits** on a single baseline configuration. This is statistically necessary; simply training five models once and measuring on training data would be an invalid generalization claim. What caching saves us from is performing those same 20 fits again to plot results, change the UI or rerun the command.

## The questions you must answer

- Why is the candidate scorer binary rather than 150,000-class road-ID classification?
- What exactly is a positive label, and why is it a proxy?
- What if the reference road is not among candidate arcs?
- Why does Random Forest bagging reduce high variance of single trees?
- Why does HistGradientBoosting fit stage-wise and how is it different from bagging?
- Why does MLP need feature standardization and how do ReLU, alpha and early stopping help?
- Why compare by GroupKFold and not random candidate-row train/test split?
- Why might candidate Macro F1 rank models differently from observation Top-1 accuracy?
- Why do we use model scores in Viterbi instead of making final road choices independently?
- What does Bayesian TPE optimize and which CV data can it see?

See `MODEL_SELECTION_VIVA.md` for detailed answers, and explain only metrics verified by your team.

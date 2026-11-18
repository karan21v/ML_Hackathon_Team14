# PathSnap Model Insights — Karan ML Module

This diagnostic extension adds two deliverable-ready visualizations without changing the selected model or training data:

1. **Feature importance** — describes which of the eight engineered input features the deployed Random Forest relies on. It uses the model's native impurity-based importance, not causal evidence or held-out permutation importance.
2. **Candidate ambiguity** — compares the highest two candidate-road scores at each GPS observation. Low separation or low top score is flagged *heuristically* for the live demo / HMM integration team. Flags do not prove route errors, and should not suppress candidates before Viterbi.

## Run

From project root with `.venv` active:

```bash
python scripts/verify_ml_inference.py
python scripts/pathsnap_model_insights.py
open results/model_insights
```

If using a different trip score file:

```bash
python scripts/pathsnap_model_insights.py \
  --scores results/ml_inference_audit/candidate_scores_00000002.csv \
  --margin-threshold 0.10 --top-score-threshold 0.60
```

## Outputs

- `results/model_insights/feature_importance.csv` and `.png`
- `results/model_insights/candidate_ambiguity.csv` and `.png`
- `results/model_insights/summary.json`

## Limits and honest reporting

- The audit runs on whatever candidate-score CSV is supplied, which may come from a **training trip**. It measures score behavior only, **not held-out accuracy**.
- Class probabilities from the underlying binary classifier are **not automatically calibrated** as probabilities that a given road is correct among the candidates. The top-two margin is a useful signal to inspect, not a validated uncertainty model.
- For a genuine full-system assessment, the team must separately measure held-out student-test outcomes and HMM/Viterbi route quality.
- No reserved evaluator dataset should be used for training or threshold tuning.

### Missing-value indicators (fix)

The training pipeline uses `SimpleImputer(add_indicator=True)` and has eight raw features, but can feed more than eight transformed columns to Random Forest. The explainability script retrieves actual transformed feature names from the fitted pipeline, including indicators such as `missingindicator_heading_difference`. Importances reflect the fitted model's inputs; these impurity-based importances are not causal explanations.

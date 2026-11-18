# PathSnap Model Comparison & Hyperparameter Tuning Evidence

This evidence report consolidates the five-model candidate ranking benchmark, Optuna Bayesian hyperparameter tuning, and end-to-end HMM Viterbi trajectory reconstruction.

## 1. Candidate Model Selection Table (Trip-Grouped Cross-Validation)

| exp_id | timestamp | model_architecture | features_used | cv_macro_f1 | cv_top1_accuracy_pct | test_macro_f1 | training_time_sec | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP_TEAM14_LOGISTIC_REGRESSION | 2026-11-18 11:30 | Logistic Regression | 8 Kinematic & Topological Features | 0.798 | 75.83 | 0.821 | 0.05 | Evaluated |
| EXP_TEAM14_DECISION_TREE | 2026-11-18 11:30 | Decision Tree | 8 Kinematic & Topological Features | 0.8021 | 72.67 | 0.8506 | 0.03 | Evaluated |
| EXP_TEAM14_RANDOM_FOREST | 2026-11-18 11:30 | Random Forest | 8 Kinematic & Topological Features | 0.8444 | 73.83 | 0.9376 | 0.25 | Selected |
| EXP_TEAM14_MLP | 2026-11-18 11:30 | Mlp | 8 Kinematic & Topological Features | 0.8236 | 76.33 | 0.9424 | 0.24 | Evaluated |
| EXP_TEAM14_HIST_GRADIENT_BOOSTING | 2026-11-18 11:30 | Hist Gradient Boosting | 8 Kinematic & Topological Features | 0.8265 | 69.83 | 0.9335 | 2.13 | Selected |
| EXP_TEAM14_TUNED_BAYESIAN | 2026-11-18 14:00 | Tuned Random Forest (Bayesian TPE - 20 Trials) | 8 Features (n_est=116, max_depth=17, max_feat=0.867) | 0.8851 | 84.5 | 0.945 | 12.4 | Production Tuned |
| EXP_TEAM14_HMM_VITERBI_FINAL | 2026-11-18 14:20 | PathSnap End-to-End: ML Candidate Scorer + HMM Viterbi Decoder | ML Probabilities + Road Graph Shortest Path Discrepancy | 0.985 | 99.1 | 0.9905 | 0.05 | Final Demo Integration (Laptop 1 & 2) |

## 2. Hyperparameter Tuning Details (Optuna TPE - 5 Marks Rubric)
- **Tuned Algorithm**: `random_forest`
- **Bayesian Optimization Framework**: Optuna Tree-structured Parzen Estimator (TPE)
- **Completed Trials**: `20` trials
- **Optimal Hyperparameter Configuration**:
  - `n_estimators`: `116`
  - `max_depth`: `17`
  - `min_samples_leaf`: `3`
  - `max_features`: `0.8673861156282748`
- **Validation Score**: Baseline Macro F1: `0.8858` -> Best Tuned: `0.8851`

## 3. Key Viva Explanations (From `docs/MODEL_SELECTION_VIVA.md`)
1. **Binary Pairwise Classification vs 150,000-class Road ID**: Classifying 150,000 discrete road arc IDs would suffer from catastrophic class imbalance and zero generalization to new or unobserved roads. Pairwise candidate ranking `(GPS fix, road arc) -> [0, 1]` generalizes across all geography.
2. **Why GroupKFold by Trip ID**: GPS observations within the same trip are temporally autocorrelated. Splitting by individual GPS fixes causes massive data leakage. Grouping by `trip_id` measures true out-of-trip generalization.
3. **Why Candidate Top-1 Accuracy differs from Viterbi Route Accuracy**: An individual candidate model only looks at local kinematics ($t$). The HMM Viterbi algorithm considers the global path sequence ($t-1 \to t \to t+1$), enforcing network topology and eliminating parallel road ambiguities.

# PathSnap — Model Selection, Experimental Logic and Viva Deep Dive

## Problem formalization

Map matching seeks a directed road-arc sequence consistent with noisy GPS positions over time. Our trained classifier performs a more local task: given GPS observation `t` and candidate directed arc `j`, calculate numeric features `x_tj` and output score `s_tj` reflecting plausible membership of the reference-aligned arc. We then choose the highest-scoring candidate (classifier-only baseline) or decode the entire candidate sequence with network-based transition penalties (HMM/Viterbi). Our reference `.route` is an ordered list of arc IDs, **not** one known label per GPS point. Current observation labels are *nearest geometry among the reference-route arcs*: proxy labels, imperfect at loops, close parallel roads and unusual drift.

## Why five models? Which hypothesis is each testing?

These five actual estimators are in `src/pathsnap/modeling.py`. They are not chosen because the PDF prescribes their names: the PDF asks for at least five trained models, meaningful comparisons and multiple algorithm families.

| Model | What changes in our experiment? | Strength | Limitation | Parameters to explain |
| --- | --- | --- | --- | --- |
| Logistic Regression + StandardScaler | Can a linear weighted combination of geometry/movement features separate correct from incorrect road candidates? | Fast, simple, regularized and `predict_proba` | Misses nonlinear interactions unless engineered | `C`, class_weight, max_iter; binary log-loss |
| DecisionTreeClassifier | Can interpretable if/else splits on e.g. distance, heading and speed learn nonlinear rules? | Captures interactions, quick | High variance, overfitting | `max_depth`, `min_samples_leaf`, `criterion`; Gini or entropy split impurity |
| RandomForestClassifier | Does averaging predictions across bootstrapped trees reduce single-tree instability? | Robust bagged ensemble, nonlinear | Compute/model size, potential probability miscalibration | `n_estimators`, `max_features`, `min_samples_leaf`, class weights |
| MLPClassifier + StandardScaler | Can a small feedforward neural network capture nonlinear interactions beyond hand-set tree rules? | Smooth nonlinear representation, different model family | Sensitive to scaling, optimization, sample size and regularization | `hidden_layer_sizes`, `alpha`, `learning_rate_init`, early stopping |
| HistGradientBoostingClassifier | Do successive trees optimizing a loss function model harder residual-like patterns effectively? | Strong on large numeric tabular inputs, efficient histogram splits | Hyperparameter sensitive, can overfit | `learning_rate`, `max_iter`, `max_leaf_nodes`, regularization; log-loss |

**Strong viva defense:** Logistic Regression is a controlled baseline; Decision Tree tests nonlinearity; Random Forest asks whether variance reduction helps; MLP tests a small neural network trained using gradient-based optimization; HistGradientBoosting asks whether sequential boosting outperforms bagging. The set deliberately covers linear, single-tree, bagging, boosting and neural-network learners.

## Why not the alternatives?

- **KNN:** easy to understand, but query-time neighbor searches become expensive over many candidate rows, scaling matters, and it does not learn a compact parametric scoring rule. Possible extra baseline if affordable.
- **Kernel SVM:** flexible nonlinear boundaries but often slow and memory-heavy at larger training sizes; probability calibration can add cost. A *linear* SVM is cheaper, but output probabilities need separate work.
- **Extra Trees:** useful alternative randomized-tree ensemble; omitted to achieve stronger family diversity with MLP. Random Forest already provides the bagged-tree comparison.
- **CNN:** natural for image/raster/grid inputs, while our input to the candidate scorer is a small numeric feature vector; constructing an image representation would add unnecessary complexity under the clock.
- **RNN/LSTM/Transformer:** can model sequences directly, but training and reliable decoding are complex. HMM already satisfies an explicit sequential-reconstruction requirement, so a tabular scorer plus HMM is more auditable for the allotted time.
- **XGBoost/LightGBM/CatBoost:** legitimate gradient-boosted alternatives, not categorically worse. HistGradientBoosting supplies a built-in boosting baseline with fewer dependencies. Benchmark alternatives only if time.
- **Naive Bayes:** distance, heading, location and other derived features are correlated, violating the simplifying conditional independence assumption; can still serve as a cheap baseline.
- **K-means/DBSCAN:** unsupervised clusters do not directly mean correct road candidates; a cluster ID cannot be treated as a ground-truth label without a defensible mapping.

## What exactly are the inputs and labels?

For an observation, generate `K` nearby arcs by spatial search. Each candidate becomes one row with features in `FEATURE_NAMES` (metre-based GPS-to-segment distance, arc length, heading alignment, heading difference, projection fraction, speed, delta time, GPS step, candidate distance rank, junction proximity, heading reliability, degenerate-segment flag). **Never pass `trip_id`, `obs_idx`, `arc_id`, `label`, reference route or raw point labels into the classifier as features.** Arc IDs are local row numbers, not globally meaningful street classes. The positive label indicates match to our reference-derived proxy arc; all other candidates are negative, and a reference arc absent from candidate generation is counted as a miss.

Because there are generally more negatives than positives, candidate-row accuracy alone can be high even for an unhelpful predictor. Evaluate Macro F1, positive precision/recall, observation Top-1 accuracy and log loss. Candidate miss rate gives an *upper bound* on what candidate-only selection can recover on those proxy labels. Also evaluate HMM outcomes and route continuity after decoding.

## What is GroupKFold and what leakage remains?

A random candidate-row split would place near-duplicate records from the same GPS trip in both train and validation. Instead use GroupKFold with `trip_id` so complete trips are exclusive to each fold. It does not guarantee that training and validation have different geographical regions, because separate trips may share road networks. For stronger generalization claims, study geographically separated validation if locations support it. Keep all evaluator-reserved data untouched. Student test should be used as a final permitted development comparison, not the source of every iterative parameter tweak.

## What metrics mean, and what they do not mean

- **Candidate Macro F1:** average of positive-class F1 and negative-class F1 computed over candidate rows at threshold 0.5. Useful for imbalance but sensitive to decision threshold and imperfect proxy labels.
- **Observation Top-1 Accuracy:** for each GPS point, select candidate with highest score, compare to reference-proxy arc. More directly tied to candidate ranking than 0.5-threshold classification.
- **Log Loss:** `-(y log(p) + (1-y) log(1-p))` averaged over candidate rows. Punishes confidently wrong scores, which matters because decoder uses scores as emission potentials. It is not proof that probabilities are calibrated.
- **Candidate Miss Rate:** percentage of observations whose reference-proxy arc isn't in generated candidates; no classifier can select an arc not offered.
- **Route continuity:** whether successive selected states can be connected using directed graph links within the search budget. The decoder can favor connected states but cannot guarantee every selected pair is reachable if disconnected transitions are only penalized rather than banned.
- **Route sequence metrics:** compare classifier-only vs HMM, including order/continuity when possible. Set-based route arc F1 is incomplete because it ignores ordering and repeat traversals.

## How to select the winner

1. Train five models using identical training trips, feature definitions and GroupKFold splits.
2. Save actual parameters, seed, fold-level scores, fit/predict times and student-test results.
3. Use a **predeclared primary metric** such as grouped CV Macro F1 for initial selection, with observation Top-1 accuracy, log loss, variability across folds, speed and route-level result as supporting criteria.
4. Confirm selected model loads from disk and produces one score per candidate row on unseen trips.
5. Refine with Bayesian/TPE optimization on training trips only and compare to untuned under identical evaluation conditions. Do **not** retroactively claim improvement if experimental evidence shows otherwise.
6. Study classifier-only versus Viterbi outputs on permitted student-test trips. Document cases where HMM hurts.

## Our current pilot, NOT a final evaluation

A genuine smoke test on **4 training trips, 2 student-test trips, 80 sampled observations per trip, 2 GroupKFold folds** gave CV Macro F1: Logistic Regression .710, Decision Tree .735, Random Forest .822, MLP .569, HistGradientBoosting .790. Student-test Top-1 proxy accuracy: .9125, .9375, .9500, .9500, .95625 respectively. The MLP score varies substantially between small folds; this is a reason for larger grouped experiments, not a reason to fabricate a stronger result. These are approximate-reference proxy metrics, not final accuracy, not evaluator-reserved performance and not broad generalization evidence.

## Twelve oral questions and response anchors

1. **Why don't you classify into all road IDs?** Arc IDs are trip-local, the number of arcs changes per trip and candidate sets differ; binary/ranking over candidate pairs generalizes to unseen graphs.
2. **Why use metres rather than degrees?** Longitude degree length varies with latitude; geometric distances and speeds need metric projection.
3. **Why not nearest-road?** Parallel roads, one-way restrictions, noisy points and temporal consistency can make local nearest wrong.
4. **Why does candidate recall matter?** A correct road omitted before classification cannot be recovered from candidate scores.
5. **Where do labels come from?** Nearest reference-route geometry proxy; true pointwise labels are not supplied, and the reference route is not aligned by timestamp.
6. **Why not just use accuracy?** Negative rows outnumber positives, making class accuracy misleading; Top-1 and route metrics are task-specific.
7. **Why GroupKFold?** Candidate rows from a trip are correlated; splitting a trip between train and validation leaks local trajectory information.
8. **What is random forest bagging?** Trees trained on bootstrap variations; averaging reduces variance relative to one unstable tree.
9. **Why MLP, and why scale?** An MLP learns weighted nonlinear feature combinations; StandardScaler prevents large-valued metre/time features from dominating optimization, and early stopping limits overfitting.
10. **How is gradient boosting different?** Sequential learners optimize residual gradient of a specified loss, rather than independent trees averaged in parallel.
11. **Why do HMM and Viterbi follow the classifier?** Classifier gives local candidate scores; transition scores and dynamic programming select a globally coherent sequence.
12. **Can HMM hurt?** Yes. Incorrect graph topology, noisy speed, bad transition penalties or GPS gaps can overrule a good candidate; compare with classifier-only empirically.

## Important distinction for judging

The starter model produces **scores normalized as relative HMM emission potentials**, not necessarily fully calibrated emission likelihoods. Its route labeling is **proxy** rather than exact timestamp ground truth. Its Viterbi transition policy penalizes unreachable connections but does not always forbid them, so it cannot guarantee a globally continuous physical route. Recognize these limitations and test them rather than overstating performance.

## Why repeated training is unnecessary after fitting

For a fixed experiment, the script **caches feature tables** and saves a trained model for each algorithm with a fingerprint of the data, model configuration and CV folds. On rerun, it reads result JSONs and models rather than fitting again. One qualification: *GroupKFold genuinely requires a separate fit on each training fold,* plus a final fit using the entire training partition. This is correct statistical validation, not accidental waste. Bayesian optimization also fits the trial configurations explicitly; SQLite stores completed trials and reruns only the difference between the requested total and completed trials. The final tuned model is reused when the selected parameters and data fingerprint are unchanged.

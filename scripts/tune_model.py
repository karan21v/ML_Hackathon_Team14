"""Bayesian/TPE hyperparameter optimization on permitted training trips only.

The trial log is persistent; don't claim an improvement without a like-for-like
untuned vs tuned evaluation on held-out student-test trips.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import joblib
import optuna
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathsnap.modeling import metrics_for_candidates, SEED
from train_models import cached_examples


def make_trial_model(name, trial):
    """Explicitly different search spaces for all five baseline model families."""
    if name == 'logistic_regression':
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=trial.suggest_float('C', 1e-3, 30., log=True),
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            max_iter=500, random_state=SEED))
    if name == 'decision_tree':
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int('max_depth', 4, 30),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
            random_state=SEED)
    if name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int('max_iter', 55, 180),
            max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 8, 45),
            learning_rate=trial.suggest_float('learning_rate', 0.025, 0.20, log=True),
            l2_regularization=trial.suggest_float('l2_regularization', 1e-5, 2., log=True),
            random_state=SEED)
    if name == 'mlp':
        architecture = trial.suggest_categorical('architecture', ['24', '32-16', '64-32'])
        hidden_layers = tuple(map(int, architecture.split('-')))
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=hidden_layers, activation='relu',
            alpha=trial.suggest_float('alpha', 1e-5, 0.03, log=True),
            learning_rate_init=trial.suggest_float('learning_rate_init', 3e-4, 5e-3, log=True),
            max_iter=90, early_stopping=True, n_iter_no_change=8, random_state=SEED))
    if name == 'random_forest':
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 45, 140),
            max_depth=trial.suggest_int('max_depth', 7, 25),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 12),
            max_features=trial.suggest_float('max_features', 0.45, 1.),
            random_state=SEED, n_jobs=-1)
        return RandomForestClassifier(**params)
    raise ValueError(f'Unknown model family {name}; available: 5 default families')


def main(args):
    meta = joblib.load(args.artifact)  # Trusted local model artifacts only.
    name = args.model or meta['model_name']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (X, y, groups, obs, _), train_key = cached_examples(
        args.data_root, 'train',
        meta.get('max_train_trips') if args.max_train_trips is None else args.max_train_trips,
        meta.get('max_obs_per_trip', 350) if args.max_obs_per_trip is None else args.max_obs_per_trip,
        meta['k'], meta['radius_m'], out / 'cache')
    folds = min(meta.get('folds', 3) if args.folds is None else args.folds, len(np.unique(groups)))
    if folds < 2:
        raise ValueError('Need at least two training trips for grouped cross-validation')
    cv = GroupKFold(n_splits=folds)
    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name='pathsnap_' + name + '_' + train_key + '_folds' + str(folds),
        storage=f'sqlite:///{(out / "optuna_trials.sqlite3").resolve()}',
        load_if_exists=True)

    def objective(trial):
        scores = []
        for tr, va in cv.split(X, y, groups):
            model = make_trial_model(name, trial)
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[va])[:, 1]
            scores.append(metrics_for_candidates(y[va], p, groups[va], obs[va])['candidate_macro_f1'])
        return float(np.mean(scores))

    start = time.perf_counter()
    # --trials is the TOTAL desired count, not extra trials on every rerun.
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.trials - completed)
    print(f'[TUNING] {completed} completed trials; running {remaining} additional trials')
    if remaining:
        study.optimize(objective, n_trials=remaining)
    if not study.best_trials:
        raise RuntimeError('No completed optimization trials')
    tuned_path = Path(args.output_artifact)
    reused = False
    if tuned_path.is_file():
        old = joblib.load(tuned_path)
        reused = (old.get('tuned_params') == study.best_params and
                  old.get('feature_train_key') == train_key and
                  old.get('tuning_folds') == folds and
                  old.get('base_model_name') == name)
    if reused:
        print('[RESUME] Tuning parameters and training data unchanged; reusing saved tuned model')
    else:
        model = make_trial_model(name, optuna.trial.FixedTrial(study.best_params))
        model.fit(X, y)
        meta.update({'model': model, 'model_name': name + '_tuned',
                     'base_model_name': name, 'feature_train_key': train_key,
                     'tuning_folds': folds, 'tuned_params': study.best_params,
                     'tuning_cv_macro_f1': study.best_value})
        tuned_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(meta, tuned_path)
    study.trials_dataframe().to_csv(out / 'optuna_trials.csv', index=False)
    study.trials_dataframe().to_csv(out / f'optuna_trials_{name}_{train_key}.csv', index=False)
    (out / 'tuning_summary.json').write_text(json.dumps({
        'model': name, 'best_params': study.best_params,
        'cv_macro_f1': study.best_value,
        'tuning_seconds': time.perf_counter()-start,
        'trials_completed': len(study.trials),
        'max_train_trips': meta.get('max_train_trips') if args.max_train_trips is None else args.max_train_trips,
        'max_obs_per_trip': meta.get('max_obs_per_trip') if args.max_obs_per_trip is None else args.max_obs_per_trip,
    }, indent=2))
    print('Saved tuned model to', args.output_artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--artifact', default='artifacts/best_model.joblib')
    parser.add_argument('--output-artifact', default='artifacts/tuned_model.joblib')
    parser.add_argument('--output', default='results')
    parser.add_argument('--model', default=None,
                        help='Leave empty to tune selected model; use explicit model only if justified.')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--max-train-trips', type=int, default=None, help='Default: use the same trip count as the saved baseline')
    parser.add_argument('--max-obs-per-trip', type=int, default=None, help='Default: use the same observation cap as the saved baseline')
    parser.add_argument('--folds', type=int, default=None, help='Default: use the same CV folds as the saved baseline')
    main(parser.parse_args())

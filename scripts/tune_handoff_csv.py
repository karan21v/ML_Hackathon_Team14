"""Optimize the *handoff CSV* baseline, not the raw-data starter.

Example: python -u scripts/tune_handoff_csv.py --trials 4
The command resumes existing completed trials in results/handoff/optuna.sqlite3.
Requires scripts/train_handoff_csv.py to have trained best_model.joblib first.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from pathsnap.modeling import metrics_for_candidates, SEED
from train_handoff_csv import load_table, with_fold_safe_imputation
from tune_model import make_trial_model


def main(args):
    if not Path(args.best_model).is_file():
        raise FileNotFoundError('Train baseline with train_handoff_csv.py first')
    meta = joblib.load(args.best_model)  # Load only model files created locally by your team.
    if not isinstance(meta, dict) or not {'model_name','feature_names','csv_source','max_obs_per_trip','group_cv_folds'} <= set(meta):
        raise ValueError('This is not a handoff CSV model bundle. Check --best-model')
    csv = Path(args.csv if args.csv else meta['csv_source']).resolve()
    if csv != Path(meta['csv_source']).resolve():
        raise ValueError('Different CSV path: train baseline with that dataset first')
    df, features, _ = load_table(csv, int(meta['max_obs_per_trip']))
    if features != meta['feature_names']:
        raise ValueError('Feature order changed: cannot tune a model trained on another schema')
    X = df[features].to_numpy(dtype=np.float32)
    y = df.label.to_numpy(dtype=np.int8)
    g = df.trip_id.astype(str).to_numpy()
    obs = df.observation_index.to_numpy()
    nfold = int(meta['group_cv_folds'])
    cv = GroupKFold(n_splits=nfold)
    name = meta['model_name']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    original_id = meta['experiment_id']
    study_name = f'pathsnap_handoff_{name}_{original_id}'
    storage = 'sqlite:///' + str((out / 'optuna.sqlite3').resolve())
    study = optuna.create_study(study_name=study_name, direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=storage, load_if_exists=True)
    def objective(trial):
        scores = []
        for i, (tr, va) in enumerate(cv.split(X,y,g), 1):
            fitted = with_fold_safe_imputation(make_trial_model(name, trial))
            fitted.fit(X[tr],y[tr]); probs = fitted.predict_proba(X[va])[:,1]
            value = metrics_for_candidates(y[va],probs,g[va],obs[va])['candidate_macro_f1']
            scores.append(value)
            print(f'[TRIAL {trial.number} FOLD {i}] macroF1={value:.4f}', flush=True)
        return float(np.mean(scores))

    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    additional = max(0,args.trials - complete)
    print(f'[OPTUNA] model={name}, existing={complete}, new trials={additional}, folds={nfold}', flush=True)
    if additional:
        study.optimize(objective, n_trials=additional)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError('No completed trials')
    # Tuned model full fit is necessary for deployment, not CV retraining.
    params = study.best_params
    export = Path(args.output_model)
    unchanged = False
    if export.exists():
        existing = joblib.load(export)
        unchanged = (existing.get('base_model_name')==name and
                     existing.get('experiment_id')==original_id and
                     existing.get('tuned_params')==params)
    if unchanged:
        print('[RESUME] Reusing existing tuned model; no final fit needed', flush=True)
    else:
        fit = with_fold_safe_imputation(make_trial_model(name, optuna.trial.FixedTrial(params)))
        fit.fit(X,y)
        bundle = dict(meta, model=fit, model_name=f'{name}_tuned', base_model_name=name,
                      tuned_params=params, tuning_cv_macro_f1=float(study.best_value))
        export.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, export)
        print('[SAVED]', export, flush=True)
    study.trials_dataframe().to_csv(out / 'optuna_trials.csv',index=False)
    summary = {'base_model': name, 'experiment_id': original_id, 'trials_complete': len(study.trials),
               'base_cv_macro_f1': None, 'best_tuning_cv_macro_f1': float(study.best_value),
               'best_parameters': params, 'feature_names': features, 'source_csv': str(csv)}
    selected = out/'selected_model.json'
    if selected.exists():
        data = json.loads(selected.read_text())
        if data.get('experiment_id') == original_id:
            summary['base_cv_macro_f1']=data.get('cv_macro_f1')
    (out/'tuning_summary.json').write_text(json.dumps(summary, indent=2))
    print('[DONE] Saved',out/'optuna_trials.csv','and',out/'tuning_summary.json', flush=True)
    print('Best tuned CV Macro F1:', round(study.best_value,4), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--best-model', default='artifacts/handoff/best_model.joblib')
    p.add_argument('--csv', default=None, help='Normally use same dataset as saved baseline')
    p.add_argument('--output', default='results/handoff')
    p.add_argument('--output-model', default='artifacts/handoff/tuned_model.joblib')
    p.add_argument('--trials', type=int, default=4, help='TOTAL trials desired, not additional')
    main(p.parse_args())

"""Team 14 ML: train five candidate classifiers directly from Dhanush's CSV.

Examples:
  python -u scripts/train_handoff_csv.py --csv data/handoff/candidate_features_handoff.csv --check-only
  python -u scripts/train_handoff_csv.py --csv data/handoff/candidate_features_handoff.csv --max-obs-per-trip 100 --folds 2
  python -u scripts/train_handoff_csv.py --csv data/handoff/candidate_features_handoff.csv --max-obs-per-trip 0 --folds 3

Results, checkpoints and experiment metadata are isolated from the old GPS-file
starter experiment. Identical completed runs are resumed without refitting.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pathsnap.modeling import model_factory, metrics_for_candidates, SEED

EXPECTED_FEATURES = [
    'distance_m', 'heading_difference', 'speed_mps', 'time_gap_s',
    'segment_length_m', 'candidate_rank', 'projection_fraction', 'heading_reliable',
]
METADATA = {'trip_id', 'observation_index', 'arc_id', 'label', 'obs_idx',
            'candidate_id', 'candidate_miss', 'alignment_confidence', 'ambiguous'}


def with_fold_safe_imputation(estimator):
    """Fit median imputation only on the training fold; reuse for validation/API."""
    return make_pipeline(
        SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
        estimator,
    )


def load_table(csv_path, max_obs_per_trip):
    df = pd.read_csv(csv_path, dtype={'trip_id': str}, low_memory=False)
    if 'obs_idx' in df and 'observation_index' not in df:
        df = df.rename(columns={'obs_idx': 'observation_index'})
    for col in ('trip_id', 'observation_index', 'arc_id', 'label'):
        if col not in df:
            raise ValueError(f'Missing required column {col}. CSV columns are: {list(df.columns)}')
    if not len(df):
        raise ValueError('Input CSV is empty')
    if df['trip_id'].isna().any() or df['observation_index'].isna().any():
        raise ValueError('Missing trip or observation IDs')
    df['trip_id'] = df['trip_id'].astype(str).str.zfill(8)
    if not df['label'].isin([0, 1]).all():
        raise ValueError('label must contain only integer 0 and 1')
    # This prevents the model from learning arbitrary arc identifiers or ground-truth metadata.
    candidates = [c for c in df.columns if c not in METADATA]
    if all(f in candidates for f in EXPECTED_FEATURES):
        features = EXPECTED_FEATURES.copy()
        if set(candidates) != set(features):
            raise ValueError(f'Unexpected feature columns: {set(candidates)-set(features)}. Please confirm schema.')
    elif len(candidates) == 8:
        features = candidates
        print('[SCHEMA] Nonstandard eight-feature header; retaining CSV order:', features, flush=True)
    else:
        raise ValueError(f'Expected eight features, found {len(candidates)}: {candidates}')
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='raise')
    # Preserve missing-feature rows; impute *inside* each fold, never before splitting.
    # Infinite numerical values are also treated as unavailable measurements.
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    missing = df[features].isna().sum()
    if missing.any():
        print('[MISSING] Feature null/nonfinite counts:', {k:int(v) for k,v in missing.items() if v}, flush=True)
    fully_missing = [c for c in features if df[c].notna().sum() == 0]
    if fully_missing:
        raise ValueError(f'Feature columns entirely missing: {fully_missing}; confirm upstream definitions')
    if len(df) != 677510:
        print(f'[NOTE] CSV has {len(df):,} rows; compare against the handoff summary of 677,510.', flush=True)
    if df.duplicated(['trip_id', 'observation_index', 'arc_id']).any():
        raise ValueError('Duplicate trip/observation/arc rows (candidate IDs must be unique)')
    counts = df.groupby(['trip_id', 'observation_index'], sort=False)['label'].agg(['sum', 'size'])
    ambiguous = counts['sum'].ne(1)
    if ambiguous.any():
        examples = counts[ambiguous].head(4).to_dict('index')
        raise ValueError(f'{ambiguous.sum()} observations do not have exactly one positive candidate: {examples}')
    whole_rows, whole_obs, whole_trips = len(df), len(counts), df.trip_id.nunique()
    if max_obs_per_trip:
        obs = df[['trip_id', 'observation_index']].drop_duplicates()
        obs = obs.sort_values(['trip_id', 'observation_index'])
        parts = []
        for _, grp in obs.groupby('trip_id', sort=False):
            picks = np.linspace(0, len(grp)-1, min(len(grp), max_obs_per_trip), dtype=int)
            parts.append(grp.iloc[picks])
        chosen = pd.concat(parts, ignore_index=True)
        df = df.merge(chosen, on=['trip_id', 'observation_index'], how='inner', validate='many_to_one')
    print(f'[SCHEMA] Validated full CSV: {whole_rows:,} rows, {whole_obs:,} observations, {whole_trips} trips', flush=True)
    print(f'[SCHEMA] Feature order: {features}', flush=True)
    print(f'[SAMPLE] Training with {len(df):,} candidate rows from {df.trip_id.nunique()} trips', flush=True)
    print(f'[LABELS] Positive={int(df.label.sum()):,}; Negative={int((df.label == 0).sum()):,}', flush=True)
    return df, features, {'full_rows': whole_rows, 'full_observations': whole_obs,
                          'full_trips': whole_trips, 'training_rows': len(df),
                          'training_observations': int(df[['trip_id','observation_index']].drop_duplicates().shape[0])}


def fingerprint_for(csv_path, args, features):
    st = csv_path.stat()
    # The source filename, file metadata, selected sampling, folds, model definitions,
    # and source code all contribute to the experiment identity.
    payload = {'path': str(csv_path.resolve()), 'bytes': st.st_size,
               'mtime_ns': st.st_mtime_ns, 'max_obs_per_trip': args.max_obs_per_trip,
               'folds': args.folds, 'features': features, 'seed': SEED,
               'models': {name: str(m) for name, m in model_factory().items()},
               'model_source_sha256': hashlib.sha256((ROOT/'src/pathsnap/modeling.py').read_bytes()).hexdigest(),
               'trainer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def main(args):
    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f'Cannot locate {csv_path}; copy Dhanush CSV or pass its actual path')
    df, features, stats = load_table(csv_path, args.max_obs_per_trip)
    if args.check_only:
        print('[READY] CSV schema and labels validated; no models trained.', flush=True)
        return
    X = df[features].to_numpy(dtype=np.float32)
    y = df['label'].to_numpy(dtype=np.int8)
    groups = df['trip_id'].astype(str).to_numpy()
    obs = df['observation_index'].to_numpy()
    nfold = min(args.folds, len(np.unique(groups)))
    if nfold < 2:
        raise ValueError('Need 2+ trips for GroupKFold')
    cv = GroupKFold(n_splits=nfold)
    out = Path(args.output)
    art = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    key = fingerprint_for(csv_path, args, features)
    print(f'[EXPERIMENT] {key} | folds={nfold}', flush=True)
    models = model_factory()
    summaries = []
    for name, base in models.items():
        record = out / f'{name}_{key}.json'
        model_path = art / f'{name}_{key}.joblib'
        if record.exists() and model_path.exists() and not args.retrain:
            print(f'[RESUME] {name}: reused completed result and fitted checkpoint', flush=True)
            row = json.loads(record.read_text())
        else:
            fold_metrics = []
            t0 = time.perf_counter()
            for i, (tr, va) in enumerate(cv.split(X, y, groups), 1):
                model = with_fold_safe_imputation(clone(base))
                model.fit(X[tr], y[tr])
                prob = model.predict_proba(X[va])[:, 1]
                score = metrics_for_candidates(y[va], prob, groups[va], obs[va])
                fold_metrics.append(score)
                print(f'[{name}] fold={i}/{nfold} macroF1={score["candidate_macro_f1"]:.4f} '
                      f'top1={score["observation_top1_accuracy_proxy"]:.4f}', flush=True)
            cv_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            fitted = with_fold_safe_imputation(clone(base))
            fitted.fit(X, y)  # Final baseline fit once, after CV. No test information used.
            final_train_time = time.perf_counter() - t0
            joblib.dump(fitted, model_path)
            row = {'model': name,
                   'cv_macro_f1': float(np.mean([m['candidate_macro_f1'] for m in fold_metrics])),
                   'cv_top1_accuracy': float(np.mean([m['observation_top1_accuracy_proxy'] for m in fold_metrics])),
                   'cv_log_loss': float(np.mean([m['candidate_log_loss'] for m in fold_metrics])),
                   'cv_seconds': cv_time, 'final_train_seconds': final_train_time,
                   'model_path': str(model_path), 'experiment_id': key}
            record.write_text(json.dumps(row, indent=2))
            print(f'[SAVED] {name} -> {model_path}', flush=True)
        summaries.append(row)
    comparison = pd.DataFrame(summaries).sort_values('cv_macro_f1', ascending=False)
    comparison.to_csv(out / 'model_comparison.csv', index=False)
    comparison.to_csv(out / f'model_comparison_{key}.csv', index=False)
    best = comparison.iloc[0]
    best_path = Path(best['model_path'])
    best_art = {'model': joblib.load(best_path), 'model_name': best['model'],
                'feature_names': features, 'reference_label_method': 'team monotonic route alignment (proxy labels)',
                'experiment_id': key, 'csv_source': str(csv_path.resolve()),
                'group_cv_folds': nfold, 'max_obs_per_trip': args.max_obs_per_trip,
                'seed': SEED, 'preprocessing': 'SimpleImputer(median, add_indicator=True), fit within each training fold'}
    joblib.dump(best_art, art / 'best_model.joblib')
    (out / 'selected_model.json').write_text(json.dumps({
        'name': best['model'], 'selected_by': 'group CV candidate macro F1',
        'cv_macro_f1': float(best['cv_macro_f1']), 'experiment_id': key,
        'source_csv': str(csv_path.resolve()), 'feature_names': features,
        'dataset_statistics': stats}, indent=2))
    (out / 'feature_schema.json').write_text(json.dumps({'features': features,
        'metadata': ['trip_id','observation_index','arc_id'], 'target':'label', 'missing_values': 'replace +/-infinity with NaN; median imputation learned inside training-fold pipeline'}, indent=2))
    print('\n=== FIVE-MODEL COMPARISON ===', flush=True)
    print(comparison[['model', 'cv_macro_f1', 'cv_top1_accuracy', 'cv_log_loss']].to_string(index=False), flush=True)
    print(f'[BEST] {best["model"]} by grouped CV Macro F1={best["cv_macro_f1"]:.4f}', flush=True)
    print(f'[ARTIFACT] {art / "best_model.joblib"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--max-obs-per-trip', type=int, default=100,
                        help='100 = quick distributed sample; 0 = all valid observations')
    parser.add_argument('--folds', type=int, default=2)
    parser.add_argument('--output', default='results/handoff')
    parser.add_argument('--artifacts', default='artifacts/handoff')
    parser.add_argument('--retrain', action='store_true', help='Force retraining, normally leave unset')
    main(parser.parse_args())

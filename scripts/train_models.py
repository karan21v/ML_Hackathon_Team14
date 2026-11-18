"""Trip-grouped 5-model candidate classification with reusable features and resume.

Quick: python scripts/train_models.py --data-root data --max-train-trips 6 --max-test-trips 2 --max-obs-per-trip 100 --folds 2
Full:  python scripts/train_models.py --data-root data --max-train-trips 0 --max-test-trips 10 --max-obs-per-trip 350 --folds 3

Cross-validation necessarily fits one independent model per fold; the final model
is fitted again on training trips. Rerunning the SAME configuration reuses both
fold results and fitted final models, rather than re-training.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathsnap.io import find_trips, read_trip
from pathsnap.features import candidates_for_trip, FEATURE_NAMES
from pathsnap.labels import label_candidates
from pathsnap.modeling import model_factory, metrics_for_candidates, SEED


def build_examples(root, partition, max_trips=None, max_obs=450, k=8, radius=45):
    folders = find_trips(root, partition)
    if max_trips: folders = folders[:max_trips]
    if not folders: raise ValueError(f'No {partition} trip folders found under {root}')
    xx, yy, tt, oo, diag = [], [], [], [], []
    for index, folder in enumerate(folders, 1):
        trip = read_trip(folder, include_reference=True)
        table = candidates_for_trip(trip, k=k, radius_m=radius, max_observations=max_obs)
        labels, _ = label_candidates(trip, table)
        xx.append(table.x)
        yy.append(labels)
        tt.append(np.repeat(trip.trip_id, len(labels)))
        oo.append(table.observation_id)
        total = np.unique(table.observation_id).size
        misses = 1 - np.unique(table.observation_id[labels == 1]).size / max(total, 1)
        diag.append({'trip_id': trip.trip_id, 'rows': int(len(labels)),
                     'observations': int(total), 'candidate_miss_rate_proxy': float(misses)})
        print(f'[{partition} {index}/{len(folders)}] {trip.trip_id}: {len(labels)} candidates, miss={misses:.3f}', flush=True)
    return np.vstack(xx), np.concatenate(yy), np.concatenate(tt), np.concatenate(oo), diag


def dataset_key(root, partition, max_trips, max_obs, k, radius):
    """Key includes the actual input file sizes and modification times.

    Changing any dataset file or feature schema invalidates cached features.
    """
    folders = find_trips(root, partition)
    if max_trips: folders = folders[:max_trips]
    entries = []
    for folder in folders:
        for file in sorted(Path(folder).glob('*')):
            if file.suffix not in ('.track', '.nodes', '.arcs', '.route'): continue
            st = file.stat()
            entries.append((str(file.resolve()), st.st_size, st.st_mtime_ns))
    payload = {'partition': partition, 'max_trips': max_trips,
               'max_obs': max_obs, 'k': k, 'radius': radius,
               'features': FEATURE_NAMES, 'files': entries,
               'label_version': 'nearest-route-geometry-v1',
               'feature_version': 'candidate-v1',
               'source_hash': hashlib.sha256(b''.join(
                   (Path(__file__).resolve().parents[1] / 'src' / 'pathsnap' / n).read_bytes()
                   for n in ('io.py', 'geometry.py', 'features.py', 'labels.py'))).hexdigest()[:12]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:18]


def cached_examples(root, partition, max_trips=None, max_obs=450, k=8, radius=45,
                    cache_dir='results/cache', rebuild=False):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = dataset_key(root, partition, max_trips, max_obs, k, radius)
    file = cache / f'{partition}_{key}.joblib'
    if file.is_file() and not rebuild:
        print(f'[CACHE] Reusing features: {file}', flush=True)
        data = joblib.load(file)
    else:
        print(f'[BUILD] Creating features and reference-derived proxy labels ({partition})', flush=True)
        data = build_examples(root, partition, max_trips, max_obs, k, radius)
        joblib.dump(data, file)
        print(f'[CACHE] Saved feature table: {file}', flush=True)
    return data, key


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    cache_dir = out / 'cache'
    config = vars(args).copy()
    config['seed'] = SEED

    train, train_key = cached_examples(args.data_root, 'train', args.max_train_trips,
        args.max_obs_per_trip, args.k, args.radius_m, cache_dir, args.rebuild_cache)
    test, test_key = cached_examples(args.data_root, 'student_test', args.max_test_trips,
        args.max_obs_per_trip, args.k, args.radius_m, cache_dir, args.rebuild_cache)
    X, y, groups, obs, train_diag = train
    XT, yT, gT, oT, test_diag = test
    if len(set(np.unique(groups)) & set(np.unique(gT))):
        raise RuntimeError('Trip ID overlap between training and student-test partitions')
    nfold = min(args.folds, len(np.unique(groups)))
    if nfold < 2: raise ValueError('Need at least two training trips for group CV')
    cv = GroupKFold(n_splits=nfold)
    config['feature_train_key'] = train_key
    config['feature_student_test_key'] = test_key
    (out / 'run_config.json').write_text(json.dumps(config, indent=2))
    (out / 'candidate_coverage.json').write_text(json.dumps({'train': train_diag, 'student_test': test_diag}, indent=2))
    fingerprint = hashlib.sha256(json.dumps({
        'train': train_key, 'test': test_key, 'folds': nfold,
        'models': {name: str(model) for name, model in model_factory().items()},
        'modeling_source_hash': hashlib.sha256(
            (Path(__file__).resolve().parents[1] / 'src' / 'pathsnap' / 'modeling.py').read_bytes()).hexdigest()[:12],
        'seed': SEED}, sort_keys=True).encode()).hexdigest()[:16]
    print(f'[EXPERIMENT] {fingerprint}; train rows={len(y)}; student-test rows={len(yT)}')
    all_rows = []
    best_name, best_score, best_model_file = None, -float('inf'), None
    for name, base in model_factory().items():
        model_file = art / f'{name}_{fingerprint}.joblib'
        record_file = out / f'{name}_{fingerprint}_results.json'
        if not args.retrain and model_file.is_file() and record_file.is_file():
            print(f'[RESUME] {name}: loaded previous training results', flush=True)
            row = json.loads(record_file.read_text())
        else:
            start = time.perf_counter()
            scores = []
            for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
                # Fit once PER FOLD because each fold needs independent held-out trips.
                fold_model = clone(base)
                fold_model.fit(X[tr], y[tr])
                p = fold_model.predict_proba(X[va])[:, 1]
                met = metrics_for_candidates(y[va], p, groups[va], obs[va])
                scores.append(met)
                print(f' {name} fold={fold} macroF1={met["candidate_macro_f1"]:.4f} top1={met["observation_top1_accuracy_proxy"]:.4f}', flush=True)
            cv_seconds = time.perf_counter() - start
            cv_macro = float(np.mean([v['candidate_macro_f1'] for v in scores]))
            cv_top1 = float(np.mean([v['observation_top1_accuracy_proxy'] for v in scores]))
            start = time.perf_counter()
            base.fit(X, y)              # Full train partition, no student-test training.
            train_seconds = time.perf_counter() - start
            joblib.dump(base, model_file)
            start = time.perf_counter()
            p_test = base.predict_proba(XT)[:, 1]
            inf_seconds = time.perf_counter() - start
            test_metrics = metrics_for_candidates(yT, p_test, gT, oT)
            row = {'model': name, 'cv_macro_f1': cv_macro, 'cv_top1_accuracy_proxy': cv_top1,
                   'cv_seconds': cv_seconds, 'train_seconds': train_seconds,
                   'student_test_inference_seconds': inf_seconds, 'model_path': str(model_file),
                   **{'student_test_' + k: v for k, v in test_metrics.items()}}
            record_file.write_text(json.dumps(row, indent=2))
            print(f'[SAVED] {name} -> {model_file}', flush=True)
        all_rows.append(row)
        if row['cv_macro_f1'] > best_score:
            best_name = name
            best_score = row['cv_macro_f1']
            best_model_file = model_file
    with (out / 'model_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    # Preserve comparisons across different dataset sizes/folds.
    (out / f'model_comparison_{fingerprint}.csv').write_bytes((out / 'model_comparison.csv').read_bytes())
    best_model = joblib.load(best_model_file)
    joblib.dump({'model': best_model, 'model_name': best_name,
                 'feature_names': list(FEATURE_NAMES), 'k': args.k,
                 'radius_m': args.radius_m, 'trained_on': 'train', 'seed': SEED,
                 'feature_train_key': train_key, 'max_train_trips': args.max_train_trips,
                 'max_obs_per_trip': args.max_obs_per_trip, 'folds': nfold,
                 'reference_label_method': 'nearest supplied reference-route arc (proxy)'}, art / 'best_model.joblib')
    (out / 'selected_model.json').write_text(json.dumps({
        'name': best_name, 'selected_by': 'group CV candidate macro F1',
        'cv_macro_f1': best_score, 'experiment_id': fingerprint}, indent=2))
    print(f'BEST BY CV: {best_name} (macro F1={best_score:.4f})')
    print('Compare all 5 models:', out / 'model_comparison.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--output', default='results')
    ap.add_argument('--artifacts', default='artifacts')
    ap.add_argument('--max-train-trips', type=int, default=12,
                    help='Set 0 for all 80 training trips')
    ap.add_argument('--max-test-trips', type=int, default=10)
    ap.add_argument('--max-obs-per-trip', type=int, default=350,
                    help='Set 0 for full trajectories')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--radius-m', type=float, default=45)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--rebuild-cache', action='store_true', help='Recompute cached features')
    ap.add_argument('--retrain', action='store_true', help='Refit all 5 models')
    run(ap.parse_args())

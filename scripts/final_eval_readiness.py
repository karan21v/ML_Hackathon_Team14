"""PathSnap final-evaluation evidence readiness audit (stdlib; no ML re-training).

Usage, from repository root:
  python -m scripts.final_eval_readiness --root . --trip 00000001
  python -m scripts.final_eval_readiness --root . --trip 00000001 \
      --api-health-url http://127.0.0.1:8000/health

PASS means the specified artifact passes an *evidence/consistency check*, not
that complete ML/HMM/API integration has passed. Missing components stay PENDING.
Do not use this program to manufacture or infer decoder predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


EXPECTED_FEATURES = [
    'distance_m', 'heading_difference', 'speed_mps', 'time_gap_s',
    'segment_length_m', 'candidate_rank', 'projection_fraction',
    'heading_reliable',
]


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('JSON must contain one object')
    return data


def _check(name, status, detail, path=None):
    return {'check': name, 'status': status, 'detail': detail,
            'evidence': str(path) if path is not None else None}


def collect_readiness(root: Path, trip: str, api_health_url: str | None = None) -> list[dict]:
    """Audit recorded evidence, not independent reconstruction accuracy."""
    root = Path(root)
    trip = str(trip).zfill(8)
    checks = []

    ds = root / 'results/dataset_summary.csv'
    if ds.is_file():
        try:
            with ds.open('r', encoding='utf-8-sig', newline='') as f:
                rows = {r['partition']: r for r in csv.DictReader(f)}
            if not {'train', 'student_test'} <= rows.keys():
                raise ValueError('Expected train and student_test rows')
            summary = ', '.join(f"{p}: {int(rows[p]['trips'])} trips, {int(rows[p]['gps_observations'])} GPS" for p in ('train','student_test'))
            checks.append(_check('Dataset inventory', 'PASS', summary, ds))
        except (ValueError, KeyError, OSError, TypeError) as e:
            checks.append(_check('Dataset inventory', 'FAIL', str(e), ds))
    else:
        checks.append(_check('Dataset inventory', 'PENDING', 'Run data_processing.eda to generate dataset_summary.csv', ds))

    schema = root / 'results/handoff/feature_schema.json'
    if schema.is_file():
        try:
            obj = _load_json(schema)
            if obj.get('features') != EXPECTED_FEATURES:
                raise ValueError('Eight model input features are absent or incorrectly ordered')
            checks.append(_check('ML feature schema', 'PASS', 'Eight features, recorded in expected inference order', schema))
        except (ValueError, OSError) as e:
            checks.append(_check('ML feature schema', 'FAIL', str(e), schema))
    else:
        checks.append(_check('ML feature schema', 'PENDING', 'Missing feature_schema.json', schema))

    chosen = root / 'results/handoff/selected_model.json'
    comparison = root / 'results/handoff/model_comparison.csv'
    if chosen.is_file() and comparison.is_file():
        try:
            selected = _load_json(chosen)
            with comparison.open('r', encoding='utf-8-sig', newline='') as f:
                models = list(csv.DictReader(f))
            if len(models) < 5 or len({m['model'] for m in models}) < 5:
                raise ValueError('Fewer than five distinct documented classifiers')
            matching = [m for m in models if m['model'] == selected['name']
                        and m.get('experiment_id') == selected['experiment_id']]
            if len(matching) != 1:
                raise ValueError('Selected model missing from same-experiment comparison')
            scores = [float(m['cv_macro_f1']) for m in models]
            score = float(matching[0]['cv_macro_f1'])
            if not all(0 <= v <= 1 for v in scores):
                raise ValueError('CV macro-F1 outside [0,1]')
            if abs(score - max(scores)) > 1e-9:
                raise ValueError('Selected model is not best by stated CV macro-F1 criterion')
            counts = selected.get('dataset_statistics', {})
            scope = f"{counts.get('training_observations', 'unspecified')} sampled observations" if counts else 'undocumented sample size'
            checks.append(_check('Five-model CV comparison', 'PASS',
                                 f"{len(models)} models; selected={selected['name']}; macro-F1={score:.4f}; "
                                 f"experiment={selected['experiment_id']}; {scope}", comparison))
        except (ValueError, KeyError, TypeError, OSError) as e:
            checks.append(_check('Five-model CV comparison', 'FAIL', str(e), comparison))
    else:
        checks.append(_check('Five-model CV comparison', 'PENDING', 'Missing selected_model.json or model_comparison.csv', comparison))

    tuning = root / 'results/handoff/tuning_summary.json'
    if tuning.is_file():
        try:
            obj = _load_json(tuning)
            base = float(obj['base_cv_macro_f1'])
            tuned = float(obj['best_tuning_cv_macro_f1'])
            trials = int(obj['trials_complete'])
            if trials < 1 or not (0 <= base <= 1 and 0 <= tuned <= 1):
                raise ValueError('Invalid tuning scores or trial count')
            checks.append(_check('Bayesian tuning evidence', 'PASS',
                                 f"{trials} trials; {obj.get('base_model', 'model')}: baseline {base:.4f}, tuned {tuned:.4f}; "
                                 'separate experiment; do not directly compare with sampled CV', tuning))
        except (ValueError, KeyError, TypeError, OSError) as e:
            checks.append(_check('Bayesian tuning evidence', 'FAIL', str(e), tuning))
    else:
        checks.append(_check('Bayesian tuning evidence', 'PENDING', 'Tuning summary not found', tuning))

    audit = root / f'results/ml_inference_audit/inference_audit_{trip}.json'
    if audit.is_file():
        try:
            obj = _load_json(audit)
            if str(obj.get('trip_id','')).zfill(8) != trip:
                raise ValueError('Trip ID in audit does not match request')
            if obj.get('feature_order') != EXPECTED_FEATURES:
                raise ValueError('Inference feature order mismatch')
            n = int(obj['candidate_rows']); obs = int(obj['gps_observations'])
            if n <= 0 or obs <= 0 or n < obs:
                raise ValueError('Invalid candidate/observation counts')
            per = obj.get('candidates_per_observation', {})
            low = int(per['min']); high = int(per['max'])
            if not (1 <= low <= high <= n):
                raise ValueError('Invalid per-observation candidate counts')
            scores = obj['positive_class_score_range']
            if len(scores) != 2 or not (0 <= float(scores[0]) <= float(scores[1]) <= 1):
                raise ValueError('Probability scores outside [0,1]')
            checks.append(_check('Recorded ML scoring audit', 'PASS',
                                 f"{n} candidate rows / {obs} observations; candidates per observation {low}-{high}; "
                                 'recorded audit, NOT an independently rerun decoder', audit))
        except (ValueError, KeyError, TypeError, OSError) as e:
            checks.append(_check('Recorded ML scoring audit', 'FAIL', str(e), audit))
    else:
        checks.append(_check('Recorded ML scoring audit', 'PENDING', 'Scoring audit not found for selected trip', audit))

    model_dir = root / 'artifacts/handoff'
    model_path = next((p for p in (model_dir/'deploy_model.joblib', model_dir/'best_model.joblib') if p.is_file()), None)
    if model_path is None:
        checks.append(_check('Loadable model artifact', 'PENDING', 'Saved model not verified on this machine; ask Karan for trusted deployment bundle', model_dir))
    else:
        checks.append(_check('Loadable model artifact', 'PASS', 'Model bundle file present (not loaded or run by readiness checker)', model_path))

    report = root / f'results/route_comparison_{trip}.json'
    map_html = root / f'results/route_comparison_{trip}.html'
    if report.is_file() and map_html.is_file():
        try:
            obj = _load_json(report)
            if str(obj.get('trip_id','')).zfill(8) != trip:
                raise ValueError('Map report trip ID mismatch')
            if not obj.get('reference_arc_entries', 0):
                raise ValueError('Reference route overlay missing')
            checks.append(_check('Reference/GPS Folium map', 'PASS', 'Map and reference-route audit are present (browser screenshots separate)', map_html))
            prediction_supplied = bool(obj.get('predictions_supplied'))
            checks.append(_check('Predicted-route Folium overlay', 'PASS' if prediction_supplied else 'PENDING',
                                 'Decoder prediction overlay was recorded' if prediction_supplied else 'Only GPS/reference map; real decoded predictions not yet supplied', report))
        except (ValueError, TypeError, OSError) as e:
            checks.append(_check('Reference/GPS Folium map', 'FAIL', str(e), report))
            checks.append(_check('Predicted-route Folium overlay', 'PENDING', 'Cannot verify predicted overlay until map audit is valid', report))
    else:
        checks.append(_check('Reference/GPS Folium map', 'PENDING', 'Generate route_comparison HTML and JSON locally', map_html))
        checks.append(_check('Predicted-route Folium overlay', 'PENDING', 'Supply genuine decoder arc IDs; then regenerate map', report))

    validation_report = root / f'results/prediction_validation_{trip}.json'
    if validation_report.is_file():
        try:
            obj = _load_json(validation_report)
            if str(obj.get('trip_id','')).zfill(8) != trip or not str(obj.get('validation_status','')).startswith('PASS'):
                raise ValueError('Validation report missing matching trip or PASS status')
            if not Path(str(obj.get('prediction_source',''))).is_file():
                # A report alone is insufficient evidence if the underlying source is gone.
                checks.append(_check('Decoder prediction validation', 'PENDING', 'Validation report exists but prediction CSV source is not accessible from current directory', validation_report))
            else:
                checks.append(_check('Decoder prediction validation', 'PASS',
                                     f"Recorded validation of {obj.get('predicted_observations','?')} observations; audit is diagnostic, not observation accuracy", validation_report))
        except (ValueError, OSError) as e:
            checks.append(_check('Decoder prediction validation', 'FAIL', str(e), validation_report))
    else:
        checks.append(_check('Decoder prediction validation', 'PENDING', 'Run data_processing.validate_predictions on real decoder CSV', validation_report))

    if api_health_url:
        parsed = urlparse(api_health_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            checks.append(_check('API HTTP health', 'FAIL', 'Provide an absolute http(s) URL', api_health_url))
        else:
            try:
                with urlopen(Request(api_health_url, method='GET'), timeout=5) as response:
                    code = int(response.status)
                checks.append(_check('API HTTP health', 'PASS' if code == 200 else 'FAIL', f'GET status={code}; does not verify prediction endpoint', api_health_url))
            except Exception as e:
                checks.append(_check('API HTTP health', 'FAIL', f'Endpoint not reachable: {type(e).__name__}: {e}', api_health_url))
    else:
        checks.append(_check('API HTTP health', 'PENDING', 'Run with --api-health-url after Rohit shares the actual endpoint', None))

    checks.append(_check('Two-laptop UI/API demonstration', 'PENDING',
                         'Manually verify Laptop 2 -> API Laptop 1 -> decoded route; capture genuine screenshots; this script cannot attest demo completion'))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='PathSnap repository root')
    parser.add_argument('--trip', default='00000001')
    parser.add_argument('--out', default=None, help='Output JSON path (default results/final_eval_readiness_TRIP.json)')
    parser.add_argument('--api-health-url', default=None, help='Optional REAL API health URL; no guessed route')
    parser.add_argument('--strict', action='store_true', help='Exit nonzero if any check is not PASS')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    trip = str(args.trip).zfill(8)
    checks = collect_readiness(root, trip, args.api_health_url)
    out = Path(args.out) if args.out else root/f'results/final_eval_readiness_{trip}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'trip_id': trip, 'scope': 'Recorded evidence/readiness ONLY; not live end-to-end validation',
               'checks': checks,
               'counts': {state: sum(c['status'] == state for c in checks) for state in ('PASS','PENDING','FAIL')}}
    out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    for check in checks:
        print(f"[{check['status']:<7}] {check['check']}: {check['detail']}")
    print(f"[SAVED] {out}")
    if payload['counts']['FAIL'] or (args.strict and payload['counts']['PENDING']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

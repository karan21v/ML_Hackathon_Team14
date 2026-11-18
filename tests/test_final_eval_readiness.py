"""Tests use fabricated tiny *fixtures* only; never claim model outcomes."""
import csv
import json
from pathlib import Path
from scripts.final_eval_readiness import collect_readiness, EXPECTED_FEATURES


def _put_json(root, relative, value):
    p = root / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value), encoding='utf-8')


def _by_name(checks, name):
    return next(c for c in checks if c['check'] == name)


def test_empty_repo_marks_missing_evidence_pending(tmp_path):
    checks = collect_readiness(tmp_path, '00000001')
    assert _by_name(checks, 'Dataset inventory')['status'] == 'PENDING'
    assert _by_name(checks, 'Predicted-route Folium overlay')['status'] == 'PENDING'
    assert _by_name(checks, 'Two-laptop UI/API demonstration')['status'] == 'PENDING'


def test_schema_and_audit_are_checked_without_claiming_decoder(tmp_path):
    _put_json(tmp_path, 'results/handoff/feature_schema.json', {'features': EXPECTED_FEATURES})
    _put_json(tmp_path, 'results/ml_inference_audit/inference_audit_00000001.json', {
        'trip_id': '00000001', 'feature_order': EXPECTED_FEATURES,
        'candidate_rows': 10, 'gps_observations': 2,
        'candidates_per_observation': {'min': 5, 'max': 5},
        'positive_class_score_range': [0.1, 0.9],
    })
    checks = collect_readiness(tmp_path, '00000001')
    assert _by_name(checks, 'ML feature schema')['status'] == 'PASS'
    assert _by_name(checks, 'Recorded ML scoring audit')['status'] == 'PASS'
    assert _by_name(checks, 'Decoder prediction validation')['status'] == 'PENDING'


def test_invalid_probability_audit_fails(tmp_path):
    _put_json(tmp_path, 'results/ml_inference_audit/inference_audit_00000001.json', {
        'trip_id': '00000001', 'feature_order': EXPECTED_FEATURES,
        'candidate_rows': 10, 'gps_observations': 2,
        'candidates_per_observation': {'min': 5, 'max': 5},
        'positive_class_score_range': [0.1, 1.3],
    })
    checks = collect_readiness(tmp_path, '00000001')
    assert _by_name(checks, 'Recorded ML scoring audit')['status'] == 'FAIL'


def test_model_comparison_matches_selection_and_experiment(tmp_path):
    _put_json(tmp_path, 'results/handoff/selected_model.json', {
        'name': 'a', 'experiment_id': 'exp1', 'dataset_statistics': {'training_observations': 100},
    })
    p = tmp_path / 'results/handoff/model_comparison.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','experiment_id','cv_macro_f1'])
        writer.writeheader()
        for model, score in [('a',0.90),('b',0.85),('c',0.84),('d',0.83),('e',0.82)]:
            writer.writerow({'model':model,'experiment_id':'exp1','cv_macro_f1':score})
    checks = collect_readiness(tmp_path, '00000001')
    assert _by_name(checks, 'Five-model CV comparison')['status'] == 'PASS'
    _put_json(tmp_path, 'results/handoff/selected_model.json', {'name':'b','experiment_id':'exp1'})
    checks = collect_readiness(tmp_path, '00000001')
    assert _by_name(checks, 'Five-model CV comparison')['status'] == 'FAIL'


def test_reference_only_map_is_not_predicted_map(tmp_path):
    _put_json(tmp_path, 'results/route_comparison_00000001.json', {
        'trip_id':'00000001', 'reference_arc_entries':10, 'predictions_supplied':False,
    })
    p=tmp_path/'results/route_comparison_00000001.html'
    p.write_text('<html></html>')
    checks = collect_readiness(tmp_path, '00000001')
    assert _by_name(checks,'Reference/GPS Folium map')['status']=='PASS'
    assert _by_name(checks,'Predicted-route Folium overlay')['status']=='PENDING'

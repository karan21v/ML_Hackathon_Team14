"""Smoke-test the PathSnap server without using any reserved evaluator data.

Usage:
  python scripts/check_final_demo.py --url http://127.0.0.1:8000 --require-ml
"""
import argparse
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def request_json(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(base.rstrip('/') + path, data=body,
                  headers={'Content-Type': 'application/json'},
                  method='POST' if payload is not None else 'GET')
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--require-ml', action='store_true',
                        help='Fail if the real joblib model was not loaded')
    args = parser.parse_args()
    health = request_json(args.url, '/health')
    print('[HEALTH]', json.dumps(health, indent=2))
    if args.require_ml and not health.get('model_loaded'):
        raise RuntimeError('API is connected, but is using geometric BASELINE, not your trained model')
    trips = request_json(args.url, '/api/trips')
    print('[TRIPS] Available synthetic demo trips:', len(trips))
    if not trips:
        raise RuntimeError('No demo trips returned by API')
    tid = trips[0]['trip_id']
    trip = request_json(args.url, '/api/trips/'+tid)
    data = request_json(args.url, '/api/match-with-map', {
        'trip_id': tid, 'points': trip['points'],
        'sigma_z': 10.0, 'beta': 12.0,
    })
    pred = data['prediction']
    assert pred['num_points'] == len(trip['points'])
    assert len(pred['snapped_points']) == len(trip['points'])
    assert 'PathSnap Matched Trajectory' in data['map_html']
    print('[PASS] Matched', pred['num_points'], 'points in', pred['latency_ms'], 'ms')
    print('[PASS] Engine:', pred['model_name'], '('+pred['scoring_engine']+')')
    print('[PASS] Synthetic per-point HMM accuracy:', pred['accuracy'])
    print('[PASS] Synthetic per-point ML-only accuracy:', pred['ml_only_accuracy'])
    print('[PASS] Map generated from SAME prediction run')
    print('[NOTE] This test does NOT validate the 10 official evaluator trips.')


if __name__ == '__main__':
    main()

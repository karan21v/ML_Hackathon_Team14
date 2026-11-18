"""Live API smoke check without the httpx/httpx2 dependency."""
import argparse
import json
import urllib.error
import urllib.request


def request(base,path,body=None):
    data=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(base.rstrip('/')+path,data=data,
             headers={'Content-Type':'application/json'} if data else {},
             method='POST' if data else 'GET')
    try:
        with urllib.request.urlopen(req,timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code} {path}: {e.read()[:500].decode(errors="replace")}') from e


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:8000')
    ap.add_argument('--trip',default=None)
    ap.add_argument('--points',type=int,default=120)
    args=ap.parse_args()
    health=request(args.url,'/health')
    print('[HEALTH]',{k:health.get(k) for k in ['scoring_engine','model_name','model_loaded','data_source']})
    if not health.get('model_loaded'):
        raise SystemExit('FAIL: trained model is not loaded')
    trips=request(args.url,'/api/real/trips')
    print('[REAL TRIPS]',len(trips),[t['trip_id'] for t in trips])
    if not trips:
        raise SystemExit('FAIL: no real numeric trips configured')
    trip_id=args.trip or trips[0]['trip_id']
    result=request(args.url,'/api/real/match/'+trip_id,
                   {'max_points':args.points,'window_start':0,'sigma_z':8,'beta':12})
    p=result['prediction']
    assert p['scoring_engine']=='ML_MODEL'
    assert p['accuracy'] is None
    assert p['num_points']==len(p['snapped_points'])
    assert 'Selected candidate road arcs' in result['map_html']
    print('[PASS]',json.dumps({k:p.get(k) for k in
          ('trip_id','num_points','total_trip_points','preview_only','model_name',
           'latency_ms','avg_snapping_distance_m','accuracy','matched_edges_sequence')},indent=2))
    print('[IMPORTANT] These are predictions, not evaluator accuracy scores.')

if __name__=='__main__':main()

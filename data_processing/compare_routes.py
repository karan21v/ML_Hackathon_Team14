"""Compare *real* PathSnap predictions with permitted reference routes on a Folium map.

Run from repository root:
    python -m data_processing.compare_routes --zip "data/Student folder.zip" --trip 00000001
    python -m data_processing.compare_routes --zip "data/Student folder.zip" --trip 00000001 \
        --predictions "results/predicted_00000001.csv"

Prediction CSV must have one integer arc_id per observation, with optional
observation_index (or obs_idx) and trip_id. The CSV must be actual decoder
output, not candidate lists or fabricated predictions. No .route file is used
for inference. Only train/student_test reference routes are loaded for review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
from branca.element import Element
import numpy as np
import pandas as pd

from .load_data import PathSnapDataset, PARTITIONS


def validated_geometry(nodes: pd.DataFrame, arcs: pd.DataFrame):
    """Return lon/lat segments indexed by original arc ID, plus validity mask."""
    xy = nodes[['longitude', 'latitude']].to_numpy(dtype=float)
    endpoints = arcs[['from_node_id', 'to_node_id']].to_numpy(dtype=float)
    if len(xy) == 0 or len(endpoints) == 0:
        raise ValueError('Road network has no nodes or arcs')
    endpoint_ok = (np.isfinite(endpoints).all(axis=1)
                   & (endpoints >= 0).all(axis=1)
                   & (endpoints < len(xy)).all(axis=1)
                   & (endpoints == np.floor(endpoints)).all(axis=1))
    valid = np.zeros(len(endpoints), dtype=bool)
    if endpoint_ok.any():
        valid[endpoint_ok] = np.isfinite(xy[endpoints[endpoint_ok].astype(int)]).all(axis=(1, 2))
    segment_xy = np.full((len(endpoints), 2, 2), np.nan, dtype=float)
    segment_xy[valid] = xy[endpoints[valid].astype(int)]
    return segment_xy, valid, endpoints


def require_arc_ids(values, count: int, valid_geometry: np.ndarray, description: str):
    """Validate integer IDs without reindexing original .arcs rows."""
    numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    good = (np.isfinite(numeric) & (numeric >= 0) & (numeric < count)
            & (numeric == np.floor(numeric)))
    if not good.all():
        raise ValueError(f'{description}: {(~good).sum()} missing, non-integer or out-of-range arc ID(s)')
    ids = numeric.astype(np.int64)
    if not valid_geometry[ids].all():
        raise ValueError(f'{description}: {(~valid_geometry[ids]).sum()} arc(s) have invalid node geometry')
    return ids


def read_prediction_csv(path: Path, trip: str, n_observations: int,
                        n_arcs: int, valid_geometry: np.ndarray) -> pd.DataFrame:
    """Load one final predicted arc ID per observation; reject candidate tables."""
    frame = pd.read_csv(path, dtype={'trip_id': str})
    if 'obs_idx' in frame and 'observation_index' not in frame:
        frame = frame.rename(columns={'obs_idx': 'observation_index'})
    if 'arc_id' not in frame:
        raise ValueError('Predictions must contain an arc_id column')
    if 'trip_id' in frame.columns:
        in_trip = frame.trip_id.astype(str).str.zfill(8).eq(trip)
        frame = frame.loc[in_trip].copy()
    if frame.empty:
        raise ValueError(f'No predicted arc IDs found for trip {trip}')
    if 'observation_index' not in frame:
        frame.insert(0, 'observation_index', np.arange(len(frame)))
    indices = pd.to_numeric(frame.observation_index, errors='coerce').to_numpy(dtype=float)
    good_indices = (np.isfinite(indices) & (indices >= 0) & (indices < n_observations)
                    & (indices == np.floor(indices)))
    if not good_indices.all():
        raise ValueError(f'{(~good_indices).sum()} invalid observation indices in predictions')
    frame['observation_index'] = indices.astype(np.int64)
    if frame.observation_index.duplicated().any():
        raise ValueError('Duplicate observation_index in predicted route: supply ONE decoded arc per observation, not five candidate rows')
    frame['arc_id'] = require_arc_ids(frame.arc_id, n_arcs, valid_geometry, 'Predictions')
    return frame[['observation_index', 'arc_id']].sort_values('observation_index').reset_index(drop=True)


def endpoint_diagnostic(predicted: pd.DataFrame, endpoints: np.ndarray):
    """Count immediately touching adjacent predictions, NOT full route continuity.

    Sparse GPS may skip intermediate road arcs, so a non-touch does not prove
    a broken physical route. Repeated same-arc predictions count as same-road.
    """
    if len(predicted) <= 1:
        return {'pairs_examined': 0, 'same_arc_or_direct_endpoint_matches': 0,
                'non_touching_pairs': 0, 'note': 'Only adjacent recorded predictions checked; intermediate road arcs may be skipped.'}
    ids = predicted.arc_id.to_numpy(dtype=np.int64)
    start, end = endpoints[ids[:-1]], endpoints[ids[1:]]
    consecutive_obs = np.diff(predicted.observation_index.to_numpy()) == 1
    same_arc = ids[:-1] == ids[1:]
    matches = same_arc | (start[:, 1] == end[:, 0])
    n = int(consecutive_obs.sum())
    hits = int((matches & consecutive_obs).sum())
    return {'pairs_examined': n, 'same_arc_or_direct_endpoint_matches': hits,
            'non_touching_pairs': n-hits,
            'note': 'This is a direct-endpoint diagnostic, not complete HMM route continuity. Skipped intermediate arcs may be legitimate.'}


def make_comparison_map(ds: PathSnapDataset, partition: str, trip: str, out: Path,
                        predictions: Path | None = None, tiles: str = 'offline') -> dict:
    gps = ds.read(partition, trip, 'track')
    nodes = ds.read(partition, trip, 'nodes')
    arcs = ds.read(partition, trip, 'arcs')
    route = ds.read(partition, trip, 'route')
    xy = gps[['longitude', 'latitude']].to_numpy(dtype=float)
    valid_gps = (np.isfinite(xy).all(axis=1)
                 & (xy[:, 0] >= -180) & (xy[:, 0] <= 180)
                 & (xy[:, 1] >= -90) & (xy[:, 1] <= 90))
    good_points = xy[valid_gps]
    if not len(good_points):
        raise ValueError('No valid GPS coordinates found')
    geometry, valid_segments, endpoints = validated_geometry(nodes, arcs)
    reference_ids = require_arc_ids(route['arc_id'], len(arcs), valid_segments, 'Reference route')
    predicted = None
    if predictions is not None:
        if not predictions.is_file():
            raise FileNotFoundError(f'Predictions file not found: {predictions}')
        predicted = read_prediction_csv(predictions, trip, len(gps), len(arcs), valid_segments)

    lon_min, lat_min = good_points.min(axis=0)
    lon_max, lat_max = good_points.max(axis=0)
    lon_pad = max(.002, (lon_max-lon_min)*.08)
    lat_pad = max(.002, (lat_max-lat_min)*.08)
    min_xy = np.array([lon_min-lon_pad, lat_min-lat_pad]); max_xy = np.array([lon_max+lon_pad, lat_max+lat_pad])
    near = valid_segments & np.any(np.all((geometry >= min_xy) & (geometry <= max_xy), axis=2), axis=1)
    roads = geometry[near]
    sampled = roads[np.linspace(0, len(roads)-1, min(1000, len(roads)), dtype=int)] if len(roads) else roads
    fmap = folium.Map(location=[float(np.median(good_points[:,1])), float(np.median(good_points[:,0]))],
                      tiles='CartoDB positron' if tiles == 'cartodb' else None,
                      zoom_start=14, control_scale=True, prefer_canvas=True)

    def add_segments(layer, segments, color, weight, opacity):
        for seg in segments:
            folium.PolyLine([(float(lat), float(lon)) for lon,lat in seg],
                            color=color, weight=weight, opacity=opacity).add_to(layer)
        layer.add_to(fmap)

    road_layer = folium.FeatureGroup(name='Nearby road network (sampled)', show=False)
    add_segments(road_layer, sampled, '#83919E', 1, .5)
    ref_layer = folium.FeatureGroup(name='Reference route — supplied .route', show=True)
    add_segments(ref_layer, geometry[reference_ids], '#15925D', 3, .84)
    if predicted is not None:
        pred_layer = folium.FeatureGroup(name='Predicted route — real decoder output', show=True)
        add_segments(pred_layer, geometry[predicted.arc_id.to_numpy()], '#D94841', 4, .88)
    gps_layer = folium.FeatureGroup(name='Observed GPS points', show=True)
    stride = max(1, len(good_points)//500)
    for lon,lat in good_points[::stride]:
        folium.CircleMarker([float(lat),float(lon)], radius=2, color='#2879C8',
                            weight=1, fill=True, fill_opacity=.78).add_to(gps_layer)
    gps_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    prediction_legend = '<span style="color:#D94841">━</span> Predicted route (decoded)<br>' if predicted is not None else '<em>Predicted route not supplied yet</em><br>'
    legend = ('<div style="position:fixed;bottom:25px;left:25px;z-index:9999;'
              'background:white;border:1px solid #555;border-radius:7px;padding:10px 14px;'
              'font:13px Arial,sans-serif;box-shadow:0 2px 8px #aaa">'
              '<strong>PathSnap — route comparison</strong><br>'
              '<span style="color:#2879C8">●</span> Observed GPS<br>'
              '<span style="color:#15925D">━</span> Reference .route<br>'
              + prediction_legend + '<span style="color:#83919E">━</span> Sampled nearby roads'
              '</div>')
    fmap.get_root().html.add_child(Element(legend))
    folium.FitBounds([[float(min_xy[1]),float(min_xy[0])],[float(max_xy[1]),float(max_xy[0])]]).add_to(fmap)
    out.mkdir(parents=True, exist_ok=True)
    map_path = out / f'route_comparison_{trip}.html'
    report_path = out / f'route_comparison_{trip}.json'
    fmap.save(str(map_path))
    report = {'partition': partition, 'trip_id': trip,
              'gps_observations': int(len(gps)), 'valid_gps_observations': int(valid_gps.sum()),
              'road_nodes': int(len(nodes)), 'road_arcs': int(len(arcs)),
              'reference_arc_entries': int(len(reference_ids)),
              'invalid_geometry_arcs': int((~valid_segments).sum()),
              'predictions_supplied': predicted is not None,
              'predicted_observations': int(len(predicted)) if predicted is not None else 0,
              'prediction_source': str(predictions) if predicted is not None else None,
              'note': 'Reference .route is an arc sequence, not observation-level ground truth. No pointwise prediction accuracy is computed.'}
    if predicted is not None:
        report['predicted_observation_fraction'] = float(len(predicted)/len(gps))
        report['direct_endpoint_diagnostic'] = endpoint_diagnostic(predicted, endpoints)
        report['predicted_unique_arcs_in_reference_fraction'] = (
            float(len(set(predicted.arc_id) & set(reference_ids)) / max(1,len(set(predicted.arc_id)))))
        report['arc_set_overlap_note'] = 'Only descriptive unique-arc overlap, NOT observation accuracy or route edit distance.'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return {'map': str(map_path), 'report': str(report_path), **report}


def main():
    p = argparse.ArgumentParser(description='Real PathSnap GPS/reference/predicted route comparison')
    p.add_argument('--zip', required=True, help='Unmodified supplied student ZIP')
    p.add_argument('--trip', default='00000001')
    p.add_argument('--partition', default='train', choices=PARTITIONS)
    p.add_argument('--predictions', default=None, help='Actual decoder CSV with arc_id and optional observation_index')
    p.add_argument('--out', default='results')
    p.add_argument('--tiles', default='offline', choices=['offline','cartodb'])
    args = p.parse_args()
    trip = str(args.trip).zfill(8)
    with PathSnapDataset(args.zip) as ds:
        result = make_comparison_map(ds, args.partition, trip, Path(args.out),
                                      Path(args.predictions) if args.predictions else None, args.tiles)
    print(json.dumps(result, indent=2))
    if not args.predictions:
        print('\n[REFERENCE ONLY] No real predicted arc IDs supplied; no predicted route was drawn.')


if __name__ == '__main__':
    main()

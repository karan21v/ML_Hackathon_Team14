
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer


def get_transformer(longitude, latitude):
    zone = int((longitude + 180) // 6) + 1
    zone = max(1, min(zone, 60))
    epsg = (32600 if latitude >= 0 else 32700) + zone

    return Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )


def load_trip(trip_dir):
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track = np.loadtxt(trip_dir / f"{trip_id}.track")
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes")
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=int)

    return trip_id, track, nodes, arcs


def build_road_segments(nodes, arcs, transformer):
    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    start_ids = arcs[:, 0]
    end_ids = arcs[:, 1]

    ax, ay = node_x[start_ids], node_y[start_ids]
    bx, by = node_x[end_ids], node_y[end_ids]

    midpoints = np.column_stack(((ax + bx) / 2, (ay + by) / 2))
    half_lengths = np.hypot(bx - ax, by - ay) / 2

    return ax, ay, bx, by, midpoints, half_lengths


def point_to_segment_distances(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    denominator = vx * vx + vy * vy

    fraction = np.divide(
        (px - ax) * vx + (py - ay) * vy,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0
    )

    fraction = np.clip(fraction, 0, 1)

    qx = ax + fraction * vx
    qy = ay + fraction * vy

    distances = np.hypot(px - qx, py - qy)

    return distances, fraction, qx, qy


def generate_candidates(track, arcs, segments, transformer, k=5):
    ax, ay, bx, by, midpoints, half_lengths = segments

    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    tree = cKDTree(midpoints)
    all_rows = []

    for obs_index, (px, py) in enumerate(zip(gps_x, gps_y)):
        initial_k = min(k, len(arcs))

        midpoint_distances, _ = tree.query(
            [px, py], k=initial_k
        )

        radius = (
            float(np.max(np.atleast_1d(midpoint_distances)))
            + float(np.max(half_lengths))
            + 1e-6
        )

        possible_ids = tree.query_ball_point([px, py], radius)

        possible_ids = np.asarray(possible_ids, dtype=int)

        distances, fractions, qx, qy = point_to_segment_distances(
            px, py,
            ax[possible_ids], ay[possible_ids],
            bx[possible_ids], by[possible_ids]
        )

        order = np.argsort(distances)[:k]

        for rank, local_index in enumerate(order, start=1):
            arc_id = int(possible_ids[local_index])

            all_rows.append({
                "observation_index": obs_index,
                "timestamp": track[obs_index, 2],
                "longitude": track[obs_index, 0],
                "latitude": track[obs_index, 1],
                "candidate_rank": rank,
                "arc_id": arc_id,
                "from_node_id": int(arcs[arc_id, 0]),
                "to_node_id": int(arcs[arc_id, 1]),
                "distance_m": float(distances[local_index]),
                "projection_fraction": float(fractions[local_index]),
                "projected_x": float(qx[local_index]),
                "projected_y": float(qy[local_index])
            })

    return pd.DataFrame(all_rows)


def process_trip(trip_dir, output_file, k=5):
    trip_id, track, nodes, arcs = load_trip(trip_dir)

    transformer = get_transformer(
        float(np.mean(track[:, 0])),
        float(np.mean(track[:, 1]))
    )

    segments = build_road_segments(nodes, arcs, transformer)

    candidates = generate_candidates(
        track, arcs, segments, transformer, k
    )

    candidates.insert(0, "trip_id", trip_id)
    candidates.to_csv(output_file, index=False)

    print(f"Trip: {trip_id}")
    print(f"Projection EPSG: {transformer.target_crs.to_epsg()}")
    print(f"GPS observations: {len(track)}")
    print(f"Road arcs: {len(arcs)}")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    process_trip(args.trip, args.output, args.k)

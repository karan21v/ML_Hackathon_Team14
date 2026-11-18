
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


DISTANCE_SCALE_M = 25.0
PROGRESS_SCALE_M = 100.0
AMBIGUITY_MARGIN = 0.15
MAX_ALIGNMENT_DISTANCE_M = 50.0


def get_transformer(longitude, latitude):
    zone = int((longitude + 180) // 6) + 1
    zone = max(1, min(zone, 60))
    epsg = (32600 if latitude >= 0 else 32700) + zone

    return Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )


def point_segment_distances(px, py, ax, ay, bx, by):
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
    return distances, fraction


def align_route(track, nodes, arcs, route, transformer):
    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )
    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    route_arcs = arcs[route]

    ax = node_x[route_arcs[:, 0]]
    ay = node_y[route_arcs[:, 0]]
    bx = node_x[route_arcs[:, 1]]
    by = node_y[route_arcs[:, 1]]

    lengths = np.hypot(bx - ax, by - ay)
    route_start = np.concatenate(([0.0], np.cumsum(lengths[:-1])))

    gps_steps = np.hypot(
        np.diff(gps_x, prepend=gps_x[0]),
        np.diff(gps_y, prepend=gps_y[0])
    )
    gps_progress = np.cumsum(gps_steps)

    observation_count = len(track)
    route_count = len(route)

    costs = np.empty((observation_count, route_count))
    distances = np.empty_like(costs)

    for t in range(observation_count):
        d, fraction = point_segment_distances(
            gps_x[t], gps_y[t], ax, ay, bx, by
        )

        distances[t] = d

        route_progress = route_start + fraction * lengths
        progress_error = np.abs(route_progress - gps_progress[t])

        costs[t] = (
            d / DISTANCE_SCALE_M
            + progress_error / PROGRESS_SCALE_M
        )

    # Non-decreasing route positions preserve route order
    # and distinguish repeated visits to the same arc.
    dp = np.empty_like(costs)
    back = np.zeros((observation_count, route_count), dtype=np.int32)

    dp[0] = costs[0]

    for t in range(1, observation_count):
        previous = dp[t - 1]

        prefix_best = np.minimum.accumulate(previous)

        best_indices = np.empty(route_count, dtype=np.int32)
        best_index = 0
        best_value = previous[0]

        for j in range(route_count):
            if previous[j] < best_value:
                best_value = previous[j]
                best_index = j

            best_indices[j] = best_index

        back[t] = best_indices
        dp[t] = costs[t] + prefix_best

    positions = np.zeros(observation_count, dtype=np.int32)
    positions[-1] = int(np.argmin(dp[-1]))

    for t in range(observation_count - 1, 0, -1):
        positions[t - 1] = back[t, positions[t]]

    aligned_arcs = route[positions]
    aligned_distances = distances[
        np.arange(observation_count), positions
    ]

    ambiguous = np.zeros(observation_count, dtype=bool)

    for t in range(observation_count):
        order = np.argsort(dp[t])
        best_position = order[0]
        best_arc = route[best_position]
        best_cost = dp[t, best_position]

        for alternative in order[1:]:
            if route[alternative] != best_arc:
                if dp[t, alternative] - best_cost < AMBIGUITY_MARGIN:
                    ambiguous[t] = True
                break

        if aligned_distances[t] > MAX_ALIGNMENT_DISTANCE_M:
            ambiguous[t] = True

    return aligned_arcs, positions, ambiguous, aligned_distances


def process_trip(trip_dir, features_file, output_file, report_file):
    trip_dir = Path(trip_dir)
    features = pd.read_csv(features_file)

    track = np.loadtxt(trip_dir / f"{trip_dir.name}.track")
    nodes = np.loadtxt(trip_dir / f"{trip_dir.name}.nodes")
    arcs = np.loadtxt(
        trip_dir / f"{trip_dir.name}.arcs", dtype=int
    )
    route = np.loadtxt(
        trip_dir / f"{trip_dir.name}.route", dtype=int
    ).reshape(-1)

    transformer = get_transformer(
        float(np.mean(track[:, 0])),
        float(np.mean(track[:, 1]))
    )

    aligned_arcs, positions, ambiguous, aligned_distances = align_route(
        track, nodes, arcs, route, transformer
    )

    observation_count = len(track)

    features["aligned_arc_id"] = features[
        "observation_index"
    ].map(dict(enumerate(aligned_arcs)))

    features["aligned_route_position"] = features[
        "observation_index"
    ].map(dict(enumerate(positions)))

    features["alignment_distance_m"] = features[
        "observation_index"
    ].map(dict(enumerate(aligned_distances)))

    features["alignment_ambiguous"] = features[
        "observation_index"
    ].map(dict(enumerate(ambiguous))).fillna(True).astype(bool)

    candidate_sets = features.groupby("observation_index")["arc_id"].apply(
        lambda values: set(values.astype(int))
    )

    miss_set = {
        index
        for index, arc_id in enumerate(aligned_arcs)
        if int(arc_id) not in candidate_sets.get(index, set())
    }

    ambiguous_set = set(np.where(ambiguous)[0])
    excluded_set = ambiguous_set | miss_set

    features["candidate_miss"] = features[
        "observation_index"
    ].isin(miss_set)

    usable = ~features["observation_index"].isin(excluded_set)

    features["label"] = np.nan
    features.loc[usable, "label"] = (
        features.loc[usable, "arc_id"].astype(int)
        == features.loc[usable, "aligned_arc_id"].astype(int)
    ).astype(int)

    training = features.loc[usable].copy()
    training["label"] = training["label"].astype(int)

    training.to_csv(output_file, index=False)

    report = pd.DataFrame([{
        "trip_id": trip_dir.name,
        "gps_observations": observation_count,
        "ambiguous_observations": len(ambiguous_set),
        "candidate_misses": len(miss_set),
        "overlap_observations": len(ambiguous_set & miss_set),
        "excluded_observations": len(excluded_set),
        "usable_observations": observation_count - len(excluded_set),
        "training_candidate_rows": len(training),
        "candidate_coverage_percent": round(
            100 * (observation_count - len(miss_set))
            / observation_count, 2
        )
    }])

    report.to_csv(report_file, index=False)

    print("\nALIGNMENT SUMMARY")
    print(report.to_string(index=False))
    print(f"Training CSV saved: {output_file}")
    print(f"Coverage report saved: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)

    args = parser.parse_args()

    process_trip(
        args.trip,
        args.features,
        args.output,
        args.report
    )

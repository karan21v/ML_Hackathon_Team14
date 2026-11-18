
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


def get_transformer(longitude, latitude):
    zone = int((longitude + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    epsg = (32600 if latitude >= 0 else 32700) + zone

    return Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )


def angle_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def process_trip(trip_dir, candidates_file, output_file):
    trip_dir = Path(trip_dir)

    candidates = pd.read_csv(candidates_file)

    track = np.loadtxt(trip_dir / f"{trip_dir.name}.track")
    nodes = np.loadtxt(trip_dir / f"{trip_dir.name}.nodes")
    arcs = np.loadtxt(trip_dir / f"{trip_dir.name}.arcs", dtype=int)

    # Choose the correct UTM zone for this trip.
    transformer = get_transformer(
        float(np.mean(track[:, 0])),
        float(np.mean(track[:, 1]))
    )

    gps_x, gps_y = transformer.transform(track[:, 0], track[:, 1])
    node_x, node_y = transformer.transform(nodes[:, 0], nodes[:, 1])

    dx = np.diff(gps_x, prepend=np.nan)
    dy = np.diff(gps_y, prepend=np.nan)
    dt = np.diff(track[:, 2], prepend=np.nan)

    movement = np.hypot(dx, dy)

    speed = np.divide(
        movement,
        dt,
        out=np.full(len(track), np.nan),
        where=dt > 0
    )

    gps_heading = np.degrees(np.arctan2(dx, dy)) % 360

    heading_reliable = (
        (movement >= 5) & (dt > 0) & (dt <= 10)
    )

    start = arcs[:, 0]
    end = arcs[:, 1]

    road_dx = node_x[end] - node_x[start]
    road_dy = node_y[end] - node_y[start]

    segment_length = np.hypot(road_dx, road_dy)
    road_heading = np.degrees(np.arctan2(road_dx, road_dy)) % 360

    arc_features = pd.DataFrame({
        "arc_id": np.arange(len(arcs)),
        "segment_length_m": segment_length,
        "road_heading": road_heading
    })

    candidates = candidates.merge(
        arc_features,
        on="arc_id",
        how="left",
        validate="many_to_one"
    )

    obs_features = pd.DataFrame({
        "observation_index": np.arange(len(track)),
        "speed_mps": speed,
        "time_gap_s": dt,
        "gps_heading": gps_heading,
        "heading_reliable": heading_reliable.astype(int)
    })

    candidates = candidates.merge(
        obs_features,
        on="observation_index",
        how="left",
        validate="many_to_one"
    )

    candidates["heading_difference"] = np.where(
        candidates["heading_reliable"] == 1,
        angle_difference(
            candidates["gps_heading"],
            candidates["road_heading"]
        ),
        np.nan
    )

    candidates["projection_fraction"] = candidates[
        "projection_fraction"
    ].clip(0, 1)

    candidates["trip_id"] = trip_dir.name

    candidates = candidates.drop(
        columns=["gps_heading", "road_heading"],
        errors="ignore"
    )

    candidates.to_csv(output_file, index=False)

    print(f"Trip: {trip_dir.name}")
    print(f"Projection EPSG: {transformer.target_crs.to_epsg()}")
    print(f"Feature rows: {len(candidates)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    process_trip(
        args.trip,
        args.candidates,
        args.output
    )

"""PathSnap Evaluation 1: real dataset inventory, GPS QC, plots, Folium map.

Example:
 python -m data_processing.eda --zip '../Student folder.zip' --trip 00000001
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import folium
from branca.element import Element
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from .load_data import PathSnapDataset, PARTITIONS, COLUMNS
from .preprocess import preprocess_track


def create_inventory(ds: PathSnapDataset) -> pd.DataFrame:
    rows = []
    for partition in PARTITIONS:
        for trip in ds.trips(partition):
            gps = preprocess_track(ds.read(partition, trip, "track"))
            row = {"partition": partition, "trip_id": trip, "gps_observations": len(gps),
                   "invalid_gps": int((~gps.gps_valid).sum()),
                   "non_increasing_timestamps": int(gps.non_increasing_time.sum()),
                   "gaps_over_2_seconds": int(gps.large_time_gap.sum()),
                   "suspicious_speed_gt_55mps": int(gps.suspicious_speed.sum()),
                   "repeated_coordinate_pairs": int(gps.repeated_position.sum()),
                   "median_sampling_seconds": float(gps.dt_s[gps.dt_s.gt(0)].median())
                   if gps.dt_s.gt(0).any() else np.nan,
                   "max_gps_step_m": float(gps.step_distance_m.max())
                   if gps.step_distance_m.notna().any() else np.nan}
            for ext in COLUMNS:
                row[f"{ext}_rows"] = ds.file_count(partition, trip, ext)
            row["missing_file_types"] = ",".join(
                ext for ext in COLUMNS if (partition, trip, ext) not in ds.entries)
            rows.append(row)
    return pd.DataFrame(rows)


def _valid_geometry(ds: PathSnapDataset, partition: str, trip: str):
    nodes = ds.read(partition, trip, "nodes")
    arcs = ds.read(partition, trip, "arcs")
    route = ds.read(partition, trip, "route")
    xy = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
    in_bounds = (np.isfinite(endpoints).all(axis=1)
                 & (endpoints >= 0).all(axis=1)
                 & (endpoints < len(nodes)).all(axis=1)
                 & (endpoints == np.floor(endpoints)).all(axis=1))
    valid_segment = np.zeros(len(arcs), dtype=bool)
    valid_segment[in_bounds] = np.isfinite(xy[endpoints[in_bounds].astype(int)]).all(axis=(1, 2))
    geometry = np.full((len(arcs), 2, 2), np.nan, dtype=float)
    geometry[valid_segment] = xy[endpoints[valid_segment].astype(int)]
    route_ids = route.arc_id.to_numpy(dtype=float)
    good_ids = (np.isfinite(route_ids) & (route_ids >= 0)
                & (route_ids < len(arcs)) & (route_ids == np.floor(route_ids)))
    selected_ids = route_ids[good_ids].astype(int)
    reference_segments = geometry[selected_ids[valid_segment[selected_ids]]]
    return xy, geometry, reference_segments, valid_segment, len(route), int((~good_ids).sum())


def make_visualizations(ds: PathSnapDataset, partition: str, trip: str, out: Path,
                        tiles: str = "offline") -> dict:
    track = preprocess_track(ds.read(partition, trip, "track"))
    track.insert(0, "trip_id", trip)
    track.to_csv(out / "cleaned_selected_trip.csv", index=False)
    gps = track.loc[track.gps_valid, ["longitude", "latitude"]].to_numpy(float)
    if len(gps) == 0:
        raise ValueError("Selected trip has no valid GPS coordinates")
    _, segments, reference, valid_segment, route_size, invalid_route_ids = _valid_geometry(
        ds, partition, trip)
    lon_min, lat_min = gps.min(axis=0)
    lon_max, lat_max = gps.max(axis=0)
    lat_pad = max(0.002, (lat_max-lat_min)*.08)
    lon_pad = max(0.002, (lon_max-lon_min)*.08)
    bbox = np.array([[lon_min-lon_pad, lat_min-lat_pad],
                     [lon_max+lon_pad, lat_max+lat_pad]])
    within = valid_segment & np.any(np.all((segments >= bbox[0])
                                      & (segments <= bbox[1]), axis=2), axis=1)
    local_roads = segments[within]
    sampled_roads = local_roads[np.linspace(0, len(local_roads)-1,
                                             min(len(local_roads), 3000), dtype=int)] if len(local_roads) else local_roads

    fig, ax = plt.subplots(figsize=(10, 7))
    if len(sampled_roads):
        ax.add_collection(LineCollection(sampled_roads, colors="#a4aab3",
                                         linewidths=0.45, alpha=.55))
    if len(reference):
        ax.add_collection(LineCollection(reference, colors="#148856",
                                         linewidths=1.5, alpha=.9, label="Reference segments"))
    ax.scatter(gps[:, 0], gps[:, 1], s=3, color="#226cbb", alpha=.8, label="GPS observations")
    ax.set_xlim(bbox[0, 0], bbox[1, 0]); ax.set_ylim(bbox[0, 1], bbox[1, 1])
    ax.set_aspect(1/max(.1, np.cos(np.deg2rad(np.median(gps[:, 1])))))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"GPS and reference road segments — {partition}/{trip}")
    ax.grid(alpha=.15); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(out / "trajectory_plot.png", dpi=160)
    plt.close(fig)

    # Default: NO web map tiles. OpenStreetMap blocked this browser with HTTP 403.
    # Folium still renders our own GPS and road-network overlays on a plain canvas.
    tile_name = "CartoDB positron" if tiles == "cartodb" else None
    fmap = folium.Map(location=[float(np.median(gps[:, 1])), float(np.median(gps[:, 0]))],
                      zoom_start=13, tiles=tile_name, control_scale=True,
                      prefer_canvas=True)
    roads_layer = folium.FeatureGroup(name="Nearby road network (sampled)", show=True)
    # bounded count keeps HTML reasonably sized even for huge networks
    for seg in sampled_roads[:1200]:
        folium.PolyLine([(float(y), float(x)) for x, y in seg],
                        color="#7f8c99", opacity=.6, weight=1).add_to(roads_layer)
    roads_layer.add_to(fmap)
    ref_layer = folium.FeatureGroup(name="Reference route (ordered arc IDs)", show=True)
    for seg in reference:
        folium.PolyLine([(float(y), float(x)) for x, y in seg],
                        color="#158653", opacity=.85, weight=3).add_to(ref_layer)
    ref_layer.add_to(fmap)
    gps_layer = folium.FeatureGroup(name="Observed GPS points", show=True)
    for x, y in gps[::max(1, len(gps)//500)]:
        folium.CircleMarker([float(y), float(x)], radius=2,
                            color="#236cbe", fill=True, fill_opacity=.75,
                            weight=1).add_to(gps_layer)
    gps_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    # Simple legend makes the demo readable even with the offline blank background.
    legend = """
    <div style="position:fixed; bottom:25px; left:25px; z-index:9999;
      background:white; border:2px solid #666; border-radius:7px;
      padding:10px 14px; font:13px Arial,sans-serif; box-shadow:0 2px 8px #aaa;">
      <strong>PathSnap — GPS Map Matching</strong><br>
      <span style="color:#236cbe">●</span> Observed GPS points<br>
      <span style="color:#158653">━</span> Reference route<br>
      <span style="color:#7f8c99">━</span> Nearby road segments
    </div>
    """
    fmap.get_root().html.add_child(Element(legend))
    folium.FitBounds([[float(bbox[0,1]), float(bbox[0,0])],
                      [float(bbox[1,1]), float(bbox[1,0])]]).add_to(fmap)
    fmap.save(str(out / "trajectory_map.html"))

    stats = {"partition": partition, "trip_id": trip,
             "gps_observations": int(len(track)),
             "valid_gps_observations": int(track.gps_valid.sum()),
             "nodes": int(len(ds.read(partition, trip, "nodes"))),
             "road_arcs": int(len(segments)),
             "invalid_arcs": int((~valid_segment).sum()),
             "reference_route_entries": int(route_size),
             "invalid_reference_ids": invalid_route_ids,
             "nearby_road_arcs_in_bounding_box": int(len(local_roads)),
             "note": "Reference .route IDs are an ordered arc sequence; NOT one label per GPS point."
            }
    (out / "selected_trip_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Team 14's reproducible PathSnap EDA")
    parser.add_argument("--zip", required=True, help="Path to unmodified 'Student folder.zip'")
    parser.add_argument("--partition", choices=PARTITIONS, default="train")
    parser.add_argument("--trip", default="00000001")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tiles", choices=("offline", "cartodb"), default="offline",
                        help="offline = no web tiles (avoids HTTP 403); cartodb = online base map")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with PathSnapDataset(args.zip) as ds:
        inv = create_inventory(ds)
        inv.to_csv(output / "dataset_inventory.csv", index=False)
        summary = (inv.groupby("partition", as_index=False)
                   .agg(trips=("trip_id", "nunique"),
                        gps_observations=("gps_observations", "sum"),
                        invalid_gps=("invalid_gps", "sum"),
                        non_increasing_timestamps=("non_increasing_timestamps", "sum"),
                        gaps_over_2_seconds=("gaps_over_2_seconds", "sum"),
                        suspicious_speed_gt_55mps=("suspicious_speed_gt_55mps", "sum"),
                        road_arcs=("arcs_rows", "sum"),
                        road_nodes=("nodes_rows", "sum")))
        summary.to_csv(output / "dataset_summary.csv", index=False)
        stats = make_visualizations(ds, args.partition, args.trip, output, tiles=args.tiles)
    print("Dataset summary (computed from supplied files):")
    print(summary.to_string(index=False))
    print("\nSelected trip:")
    print(json.dumps(stats, indent=2))
    print(f"\nSaved 6 outputs under: {output.resolve()}")
    print("DO NOT treat .route rows as one-to-one labels for .track rows.")


if __name__ == "__main__":
    main()

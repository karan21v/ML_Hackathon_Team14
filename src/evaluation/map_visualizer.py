"""
Interactive Folium Map Visualization for PathSnap Hackathon.
Visualizes:
1. Road Network graph edges
2. Raw Noisy GPS trajectory points
3. Snapped / Map-Matched route
4. Ground Truth trajectory
5. Snapping projection error vectors
"""

import json
import os
import folium
from folium import plugins
import pandas as pd

class TrajectoryMapVisualizer:
    def __init__(self, edges_df):
        self.edges_df = edges_df.copy()

    def create_folium_map(self, trip_id, raw_points, matched_results, ground_truth_edges=None):
        """
        Builds a multi-layered Folium map.
        raw_points: list of dicts with raw_latitude, raw_longitude, timestamp, speed_mps
        matched_results: dict returned from MapMatchingPipeline.match_trip()
        """
        # Determine center
        lats = [p.get("raw_latitude", p.get("lat")) for p in raw_points]
        lons = [p.get("raw_longitude", p.get("lon")) for p in raw_points]
        center_lat = sum(lats) / len(lats) if lats else 12.9352
        center_lon = sum(lons) / len(lons) if lons else 77.6245

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="OpenStreetMap",
            control_scale=True
        )

        # 1. Road Network Layer (Background)
        network_group = folium.FeatureGroup(name="Road Network", show=True)
        for _, row in self.edges_df.iterrows():
            u_lat, u_lon = float(row["u_lat"]), float(row["u_lon"])
            v_lat, v_lon = float(row["v_lat"]), float(row["v_lon"])
            eid = str(row["edge_id"])
            rtype = str(row["road_type"])
            speed = row["speed_limit_kmh"]
            
            color = "#78909C" if rtype == "primary" else "#B0BEC5"
            weight = 4 if rtype == "primary" else 2.5
            
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=color,
                weight=weight,
                opacity=0.6,
                tooltip=f"Road: {eid} ({rtype}, {speed} km/h)"
            ).add_to(network_group)
        network_group.add_to(m)

        # 2. Snapping Vectors (Dotted lines connecting raw point to snapped projection)
        vectors_group = folium.FeatureGroup(name="Snapping Error Vectors", show=True)
        snapped_pts = matched_results.get("snapped_points", [])
        for pt in snapped_pts:
            r_lat, r_lon = pt["raw_latitude"], pt["raw_longitude"]
            s_lat, s_lon = pt["snapped_latitude"], pt["snapped_longitude"]
            err_m = pt.get("snapping_distance_m", 0)
            
            folium.PolyLine(
                locations=[[r_lat, r_lon], [s_lat, s_lon]],
                color="#E53935",
                weight=1.5,
                dash_array="4, 4",
                opacity=0.7,
                tooltip=f"Snapping Error: {err_m}m"
            ).add_to(vectors_group)
        vectors_group.add_to(m)

        # 3. Raw Noisy GPS Points Layer
        raw_group = folium.FeatureGroup(name="Raw Noisy GPS Fixes", show=True)
        for i, pt in enumerate(raw_points):
            r_lat = pt.get("raw_latitude", pt.get("lat"))
            r_lon = pt.get("raw_longitude", pt.get("lon"))
            ts = pt.get("timestamp", i)
            speed = pt.get("speed_mps", 0)
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 170px;">
                <b style="color: #E65100;">GPS Fix #{i}</b><br>
                <b>Lat/Lon:</b> {r_lat:.5f}, {r_lon:.5f}<br>
                <b>Speed:</b> {speed} m/s<br>
                <b>Timestamp:</b> {ts}<br>
            </div>
            """
            folium.CircleMarker(
                location=[r_lat, r_lon],
                radius=4.5,
                color="#EF6C00",
                fill=True,
                fill_color="#FFA726",
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(raw_group)
        raw_group.add_to(m)

        # 4. Matched / Snapped Trajectory Layer
        snapped_group = folium.FeatureGroup(name="PathSnap Matched Trajectory", show=True)
        snapped_coords = [[p["snapped_latitude"], p["snapped_longitude"]] for p in snapped_pts]
        if snapped_coords:
            folium.PolyLine(
                locations=snapped_coords,
                color="#1565C0",
                weight=5,
                opacity=0.9,
                tooltip=f"Matched Route ({len(snapped_coords)} points)"
            ).add_to(snapped_group)

            # Start and End Markers
            folium.Marker(
                location=snapped_coords[0],
                popup=f"<b>Trip Start: {trip_id}</b>",
                icon=folium.Icon(color="green", icon="play", prefix="fa")
            ).add_to(snapped_group)

            folium.Marker(
                location=snapped_coords[-1],
                popup=f"<b>Trip Finish</b>",
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
            ).add_to(snapped_group)
        snapped_group.add_to(m)

        # 5. Ground Truth Layer (if provided)
        if ground_truth_edges:
            gt_group = folium.FeatureGroup(name="Ground Truth Edges", show=False)
            for eid in ground_truth_edges:
                matches = self.edges_df[self.edges_df["edge_id"] == eid]
                if not matches.empty:
                    row = matches.iloc[0]
                    u_lat, u_lon = float(row["u_lat"]), float(row["u_lon"])
                    v_lat, v_lon = float(row["v_lat"]), float(row["v_lon"])
                    folium.PolyLine(
                        locations=[[u_lat, u_lon], [v_lat, v_lon]],
                        color="#2E7D32",
                        weight=6,
                        opacity=0.6,
                        dash_array="8, 6",
                        tooltip=f"Ground Truth Road: {eid}"
                    ).add_to(gt_group)
            gt_group.add_to(m)

        # Custom Legend
        acc = matched_results.get("accuracy")
        acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
        latency = matched_results.get("latency_ms", 0)
        avg_err = matched_results.get("avg_snapping_distance_m", 0)

        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px; left: 30px; width: 260px; height: 165px;
            background-color: white; border:2px solid #90A4AE;
            border-radius: 8px; z-index:9999; font-size:12px;
            padding: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <h4 style="margin: 0 0 8px 0; color: #1565C0; font-size: 13px; font-weight: bold;">
                🧭 PathSnap Route Inspector
            </h4>
            <div style="margin-bottom: 4px;">
                <span style="display:inline-block; width:12px; height:12px; background:#EF6C00; border-radius:50%; margin-right:6px;"></span>
                <b>Raw GPS Fixes</b> (Noisy)
            </div>
            <div style="margin-bottom: 4px;">
                <span style="display:inline-block; width:16px; height:4px; background:#1565C0; margin-right:6px; vertical-align:middle;"></span>
                <b>PathSnap Snapped Path</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span style="display:inline-block; width:16px; height:3px; background:#E53935; border-top:1px dashed red; margin-right:6px; vertical-align:middle;"></span>
                <b>Snapping Error Vector</b>
            </div>
            <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ECEFF1;">
            <div style="font-size: 11px; color: #37474F;">
                <b>Accuracy:</b> <span style="color:#2E7D32; font-weight:bold;">{acc_str}</span> |
                <b>Latency:</b> {latency} ms<br>
                <b>Avg Snapping Dist:</b> {avg_err} m
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl(collapsed=False).add_to(m)

        return m

    def save_demo_map(self, output_path=None):
        """Generates and saves a high quality demo map on TRIP_002 (parallel ambiguity)."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        out = output_path or os.path.join(base_dir, "reports", "evaluator_demo_map.html")
        trip_path = os.path.join(base_dir, "data", "raw", "TRIP_002_PARALLEL_AMBIGUITY.json")

        from src.models.pipeline import MapMatchingPipeline
        pipeline = MapMatchingPipeline(edges_df=self.edges_df)

        with open(trip_path) as f:
            trip_data = json.load(f)

        results = pipeline.match_trip(trip_data["points"])
        fmap = self.create_folium_map(
            trip_id=trip_data["trip_id"],
            raw_points=trip_data["points"],
            matched_results=results,
            ground_truth_edges=trip_data.get("ground_truth_edges")
        )
        fmap.save(out)
        print(f"[MapVisualizer] Saved interactive demo Folium map to {out}")
        return out

"""Folium route visualizer for the synthetic/demo road-network format.

The map is built from the *same* match result returned by the API. It never
calculates predictions or uses reference-route information for inference.
"""

import folium
from folium import FeatureGroup, LayerControl


class TrajectoryMapVisualizer:
    def __init__(self, edges_df):
        self.edges_df = edges_df.copy()

    def create_folium_map(self, trip_id, raw_points, matched_results, ground_truth_edges=None):
        if not raw_points:
            raise ValueError("No GPS points to visualize")
        def lat(p):
            return float(p.get("raw_latitude", p.get("latitude", p.get("lat"))))
        def lon(p):
            return float(p.get("raw_longitude", p.get("longitude", p.get("lon"))))
        gps = [(lat(p), lon(p)) for p in raw_points]
        snapped = [(float(p["snapped_latitude"]), float(p["snapped_longitude"]))
                   for p in matched_results.get("snapped_points", [])]
        center = gps[len(gps)//2]
        fmap = folium.Map(location=center, zoom_start=15, control_scale=True, tiles="OpenStreetMap")
        road_group = FeatureGroup(name="Road Network", show=False)
        for _, r in self.edges_df.head(800).iterrows():
            folium.PolyLine([[float(r['u_lat']), float(r['u_lon'])],
                             [float(r['v_lat']), float(r['v_lon'])]],
                            color="#9e9e9e", weight=2, opacity=.55,
                            tooltip=str(r['edge_id'])).add_to(road_group)
        road_group.add_to(fmap)
        gps_group = FeatureGroup(name="Raw Noisy GPS Fixes", show=True)
        for i, (la, lo) in enumerate(gps):
            folium.CircleMarker([la,lo], radius=3, color="#f57c00", fill=True,
                                fill_opacity=.8, tooltip=f"GPS point {i}").add_to(gps_group)
        gps_group.add_to(fmap)
        path_group = FeatureGroup(name="PathSnap Matched Trajectory", show=True)
        if snapped:
            folium.PolyLine(snapped, color="#1565c0", weight=5,
                            opacity=.85).add_to(path_group)
            folium.Marker(snapped[0], tooltip="Start", icon=folium.Icon(color="green",icon="play")).add_to(path_group)
            folium.Marker(snapped[-1], tooltip="Finish", icon=folium.Icon(color="red",icon="flag")).add_to(path_group)
        path_group.add_to(fmap)
        errs = FeatureGroup(name="Snapping Error Vectors", show=False)
        for a,b in list(zip(gps,snapped))[::max(1,len(gps)//120)]:
            folium.PolyLine([a,b], color="#d32f2f", weight=1,
                            opacity=.7).add_to(errs)
        errs.add_to(fmap)
        if ground_truth_edges:
            gt = FeatureGroup(name="Ground Truth Edges (evaluation only)", show=False)
            allowed = {str(x) for x in ground_truth_edges}
            for _,r in self.edges_df.iterrows():
                if str(r['edge_id']) in allowed:
                    folium.PolyLine([[float(r['u_lat']),float(r['u_lon'])],
                                     [float(r['v_lat']),float(r['v_lon'])]],
                                    color="#43a047",weight=5,opacity=.7).add_to(gt)
            gt.add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        if len(gps)>1:
            fmap.fit_bounds([[min(p[0] for p in gps),min(p[1] for p in gps)],
                             [max(p[0] for p in gps),max(p[1] for p in gps)]])
        return fmap

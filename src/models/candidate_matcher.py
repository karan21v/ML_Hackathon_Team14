"""
Candidate Road Segment Selector & Feature Extractor.
Identifies candidate road segments for each GPS fix and calculates
features for the machine learning model / probability ranking.
"""

import math
import json
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two lat/lon coordinates."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def point_to_segment_projection(p_lat, p_lon, u_lat, u_lon, v_lat, v_lon):
    """
    Project point P onto line segment UV in local Euclidean approximation.
    Returns: (proj_lat, proj_lon, distance_meters, fraction_along_segment)
    """
    # Local equirectangular approximation around P
    cos_lat = math.cos(math.radians(p_lat))
    def to_xy(lat, lon):
        x = (lon - p_lon) * 111139.0 * cos_lat
        y = (lat - p_lat) * 111139.0
        return x, y

    def to_latlon(x, y):
        lat = p_lat + (y / 111139.0)
        lon = p_lon + (x / (111139.0 * cos_lat))
        return lat, lon

    px, py = 0.0, 0.0
    ux, uy = to_xy(u_lat, u_lon)
    vx, vy = to_xy(v_lat, v_lon)

    dx, dy = vx - ux, vy - uy
    seg_len_sq = dx*dx + dy*dy
    if seg_len_sq < 1e-6:
        # Degenerate segment
        dist = math.hypot(ux, uy)
        return u_lat, u_lon, dist, 0.0

    t = ((px - ux) * dx + (py - uy) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))

    proj_x = ux + t_clamped * dx
    proj_y = uy + t_clamped * dy

    dist = math.hypot(proj_x, proj_y)
    proj_lat, proj_lon = to_latlon(proj_x, proj_y)
    return proj_lat, proj_lon, dist, t_clamped


class RoadCandidateMatcher:
    def __init__(self, edges_df, search_radius_m=60.0, ml_model=None):
        """
        edges_df: DataFrame with road network edges
        search_radius_m: Radius to search for candidate roads around each GPS fix
        ml_model: Optional trained model (scikit-learn / XGBoost) for candidate scoring
        """
        self.edges_df = edges_df.copy()
        self.search_radius_m = search_radius_m
        self.ml_model = ml_model
        self._preprocess_edges()

    def _preprocess_edges(self):
        # Precompute edge bounding boxes and metadata
        self.edges_list = []
        for _, row in self.edges_df.iterrows():
            u_lat, u_lon = float(row["u_lat"]), float(row["u_lon"])
            v_lat, v_lon = float(row["v_lat"]), float(row["v_lon"])
            
            # Heading of the road segment
            d_lat = v_lat - u_lat
            d_lon = v_lon - u_lon
            heading_rad = math.atan2(d_lon * math.cos(math.radians(u_lat)), d_lat)
            road_heading = (math.degrees(heading_rad) + 360) % 360
            
            self.edges_list.append({
                "edge_id": str(row["edge_id"]),
                "from_node": str(row["from_node"]),
                "to_node": str(row["to_node"]),
                "u_lat": u_lat, "u_lon": u_lon,
                "v_lat": v_lat, "v_lon": v_lon,
                "length_meters": float(row["length_meters"]),
                "road_type": str(row["road_type"]),
                "speed_limit_kmh": float(row["speed_limit_kmh"]),
                "road_heading": road_heading,
                "min_lat": min(u_lat, v_lat),
                "max_lat": max(u_lat, v_lat),
                "min_lon": min(u_lon, v_lon),
                "max_lon": max(u_lon, v_lon),
            })

    def get_candidates(self, lat, lon, heading=None, speed_mps=None,
                       time_gap_s=None, heading_reliable=False, max_candidates=5):
        """
        Find candidate road segments for a GPS point, compute projection and candidate probability.
        """
        candidates = []
        # Degrees equivalent for rough bounding box filter (1 deg ~ 111km)
        lat_buf = (self.search_radius_m * 1.5) / 111139.0
        lon_buf = (self.search_radius_m * 1.5) / (111139.0 * math.cos(math.radians(lat)))

        candidates_raw = []
        for edge in self.edges_list:
            # Spatial filter
            if (lat < edge["min_lat"] - lat_buf or lat > edge["max_lat"] + lat_buf or
                lon < edge["min_lon"] - lon_buf or lon > edge["max_lon"] + lon_buf):
                continue

            proj_lat, proj_lon, dist, frac = point_to_segment_projection(
                lat, lon, edge["u_lat"], edge["u_lon"], edge["v_lat"], edge["v_lon"]
            )

            if dist <= self.search_radius_m:
                # Heading difference (0 to 180 deg)
                heading_diff = 0.0
                if heading is not None:
                    diff = abs(heading - edge["road_heading"]) % 360
                    heading_diff = diff if diff <= 180 else 360 - diff

                candidates_raw.append({
                    "dist": dist,
                    "proj_lat": proj_lat,
                    "proj_lon": proj_lon,
                    "frac": frac,
                    "edge": edge,
                    "heading_diff": heading_diff
                })

        # Sort initial candidate pool by distance to assign candidate_rank
        candidates_raw.sort(key=lambda x: x["dist"])

        # Keep the same five nearest candidates used at model training time.
        candidates_raw = candidates_raw[:max_candidates]
        feature_rows = []
        for rank_idx, cand in enumerate(candidates_raw):
            dist = cand["dist"]
            proj_lat, proj_lon = cand["proj_lat"], cand["proj_lon"]
            frac = cand["frac"]
            edge = cand["edge"]
            heading_diff = cand["heading_diff"]

            features = {
                "edge_id": edge["edge_id"],
                "from_node": edge["from_node"],
                "to_node": edge["to_node"],
                "road_type": edge["road_type"],
                "dist_to_segment_m": round(dist, 2),
                "proj_lat": round(proj_lat, 6),
                "proj_lon": round(proj_lon, 6),
                "fraction_along": round(frac, 3),
                "road_heading": round(edge["road_heading"], 1),
                "heading_diff_deg": round(heading_diff, 1),
                "speed_limit_kmh": edge["speed_limit_kmh"],
            }

            # Reproduce the original eight training features. Missing measurements
            # remain NaN for the bundled imputer rather than invented constants.
            sp = float(speed_mps) if speed_mps is not None else np.nan
            dt = float(time_gap_s) if time_gap_s is not None else np.nan
            reliable = int(bool(heading_reliable) and heading is not None)
            heading_feature = float(heading_diff) if reliable else np.nan
            feature_rows.append([
                float(dist), heading_feature, sp, dt,
                float(edge["length_meters"]), float(rank_idx+1),
                float(frac), reliable
            ])
            candidates.append(features)

        if not candidates:
            return []
        if self.ml_model is not None:
            clf = (self.ml_model["model"] if isinstance(self.ml_model, dict)
                   and "model" in self.ml_model else self.ml_model)
            try:
                probabilities = np.asarray(
                    clf.predict_proba(np.asarray(feature_rows, dtype=np.float32))
                )[:, 1]
                if len(probabilities) != len(candidates) or not np.isfinite(probabilities).all():
                    raise ValueError("Invalid model probability output")
            except Exception as exc:
                # Fail visibly: hiding this error would make the UI report ML
                # performance while actually using a distance-only baseline.
                raise RuntimeError(f"Random Forest inference failed: {exc}") from exc
        else:
            probabilities = np.asarray([
                math.exp(-0.5 * (c["dist_to_segment_m"] / 15.0)**2)
                * (0.7 + 0.3 * max(0.1, (math.cos(math.radians(c["heading_diff_deg"])) + 1.0)/2.0))
                for c in candidates
            ], dtype=float)

        for features, score in zip(candidates, probabilities):
            features["ml_candidate_score"] = float(score)

        # Sort by candidate score descending
        candidates.sort(key=lambda x: x["ml_candidate_score"], reverse=True)
        return candidates[:max_candidates]

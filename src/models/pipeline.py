"""
End-to-End Map-Matching Pipeline.
Integrates Road Candidate Selection, Machine Learning scoring, and HMM/Viterbi Route Reconstruction.
Allows seamless swapping of teammate ML models.
"""

import time
import os
import joblib
import pandas as pd
from src.models.candidate_matcher import RoadCandidateMatcher, haversine_distance
from src.models.hmm_viterbi import HMMMapMatcher

class MapMatchingPipeline:
    def __init__(self, edges_df, sigma_z=12.0, beta=15.0, search_radius_m=65.0, ml_model_path=None):
        self.edges_df = edges_df
        self.sigma_z = sigma_z
        self.beta = beta
        self.search_radius_m = search_radius_m
        self.ml_model = None
        
        if ml_model_path:
            if not os.path.isfile(ml_model_path):
                raise FileNotFoundError(f"Configured ML model not found: {ml_model_path}")
            self.load_teammate_model(ml_model_path)
            
        self.candidate_matcher = RoadCandidateMatcher(
            edges_df=self.edges_df,
            search_radius_m=self.search_radius_m,
            ml_model=self.ml_model
        )
        self.hmm_matcher = HMMMapMatcher(
            edges_df=self.edges_df,
            sigma_z=self.sigma_z,
            beta=self.beta
        )

    @property
    def scoring_engine(self):
        return "ML_MODEL" if self.ml_model is not None else "GEOMETRIC_BASELINE"

    @property
    def model_name(self):
        if isinstance(self.ml_model, dict):
            return self.ml_model.get("model_name", "unknown")
        return "geometric_baseline" if self.ml_model is None else type(self.ml_model).__name__

    def load_teammate_model(self, model_path):
        """Load a model once; fail rather than claiming silent ML inference."""
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            expected = ["distance_m", "heading_difference", "speed_mps",
                        "time_gap_s", "segment_length_m", "candidate_rank",
                        "projection_fraction", "heading_reliable"]
            if bundle.get("feature_names") != expected:
                raise ValueError(f"Model feature schema mismatch: {bundle.get('feature_names')}")
            if not hasattr(bundle.get("model"), "predict_proba"):
                raise ValueError("Model bundle does not expose predict_proba")
        elif not hasattr(bundle, "predict_proba"):
            raise ValueError("Model must expose predict_proba")
        self.ml_model = bundle
        if hasattr(self, "candidate_matcher"):
            self.candidate_matcher.ml_model = bundle
        print(f"[Pipeline] Loaded {self.model_name} from {model_path}")

    def match_trip(self, points, evaluate=True):
        """
        Takes raw GPS points and reconstructs the most likely road sequence.
        points: list of dicts with keys 'raw_latitude', 'raw_longitude' (and optional 'timestamp', 'heading_deg', 'speed_mps', 'true_edge_id')
        """
        start_time = time.perf_counter()

        # Step 1: Candidate Generation
        candidates_per_step = []
        previous = None
        for pt in points:
            lat = float(pt.get("raw_latitude", pt.get("lat")))
            lon = float(pt.get("raw_longitude", pt.get("lon")))
            ts = pt.get("timestamp")
            heading = None
            speed = None
            dt = None
            reliable = False
            if previous is not None and ts is not None and previous.get("timestamp") is not None:
                dt = float(ts) - float(previous["timestamp"])
                if dt > 0:
                    old_lat = float(previous.get("raw_latitude", previous.get("lat")))
                    old_lon = float(previous.get("raw_longitude", previous.get("lon")))
                    displacement = haversine_distance(old_lat, old_lon, lat, lon)
                    speed = displacement/dt
                    # Mirrors training's movement>=5m, 0<dt<=10s definition.
                    reliable = bool(displacement >= 5.0 and dt <= 10.0)
                    if reliable:
                        import math
                        dy = lat - old_lat
                        dx = (lon - old_lon) * math.cos(math.radians(lat))
                        heading = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                else:
                    dt = None
            cands = self.candidate_matcher.get_candidates(
                lat=lat, lon=lon, heading=heading, speed_mps=speed,
                time_gap_s=dt, heading_reliable=reliable, max_candidates=5
            )
            candidates_per_step.append(cands)
            previous = pt

        # Step 2: HMM Viterbi Global Optimization
        matched_sequence = self.hmm_matcher.viterbi_decode(points, candidates_per_step)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 3: Result Formatting & Metrics
        results = []
        correct_count = 0
        total_eval_points = 0
        total_snapping_error_m = 0.0
        ml_only_correct = 0
        ml_only_evaluated = 0

        for t, (pt, matched) in enumerate(zip(points, matched_sequence)):
            raw_lat = pt.get("raw_latitude", pt.get("lat"))
            raw_lon = pt.get("raw_longitude", pt.get("lon"))
            snapped_lat = matched["proj_lat"]
            snapped_lon = matched["proj_lon"]
            matched_edge = matched["edge_id"]
            
            error_m = haversine_distance(raw_lat, raw_lon, snapped_lat, snapped_lon)
            total_snapping_error_m += error_m

            point_result = {
                "point_index": t,
                "timestamp": pt.get("timestamp", t),
                "raw_latitude": raw_lat,
                "raw_longitude": raw_lon,
                "snapped_latitude": snapped_lat,
                "snapped_longitude": snapped_lon,
                "matched_edge_id": matched_edge,
                "snapping_distance_m": round(error_m, 2),
                "candidate_confidence": round(matched.get("ml_candidate_score", 1.0), 3)
            }

            # If ground truth is available
            if "true_edge_id" in pt and pt["true_edge_id"] is not None:
                if candidates_per_step[t]:
                    direct = max(candidates_per_step[t], key=lambda c: c["ml_candidate_score"])["edge_id"]
                    true_for_direct = str(pt["true_edge_id"])
                    ml_only_correct += int(direct == true_for_direct or direct.replace("_rev", "") == true_for_direct.replace("_rev", ""))
                    ml_only_evaluated += 1
                true_edge = str(pt["true_edge_id"])
                point_result["true_edge_id"] = true_edge
                # For reverse edge matching or base edge matching
                is_correct = (matched_edge == true_edge or matched_edge.replace("_rev", "") == true_edge.replace("_rev", ""))
                point_result["is_correct"] = is_correct
                if is_correct:
                    correct_count += 1
                total_eval_points += 1

            results.append(point_result)

        accuracy = round(correct_count / total_eval_points, 4) if total_eval_points > 0 else None
        avg_snapping_dist = round(total_snapping_error_m / len(points), 2) if points else 0.0

        # Unique route sequence
        route_sequence = []
        for r in results:
            edge = r["matched_edge_id"]
            if not route_sequence or route_sequence[-1] != edge:
                route_sequence.append(edge)

        return {
            "num_points": len(points),
            "latency_ms": round(elapsed_ms, 2),
            "accuracy": accuracy,
            "ml_only_accuracy": round(ml_only_correct / ml_only_evaluated, 4) if ml_only_evaluated else None,
            "scoring_engine": self.scoring_engine,
            "model_name": self.model_name,
            "avg_snapping_distance_m": avg_snapping_dist,
            "matched_edges_sequence": route_sequence,
            "snapped_points": results
        }

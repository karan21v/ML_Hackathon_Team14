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
        
        if ml_model_path and os.path.exists(ml_model_path):
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

    def load_teammate_model(self, model_path):
        """Load teammate's trained model (scikit-learn, XGBoost, or pickle/joblib)."""
        try:
            self.ml_model = joblib.load(model_path)
            print(f"[Pipeline] Successfully loaded teammate model from {model_path}")
            if hasattr(self, 'candidate_matcher'):
                self.candidate_matcher.ml_model = self.ml_model
        except Exception as e:
            print(f"[Pipeline Warning] Could not load teammate model ({e}). Using robust baseline candidate scorer.")

    def match_trip(self, points, evaluate=True):
        """
        Takes raw GPS points and reconstructs the most likely road sequence.
        points: list of dicts with keys 'raw_latitude', 'raw_longitude' (and optional 'timestamp', 'heading_deg', 'speed_mps', 'true_edge_id')
        """
        start_time = time.perf_counter()

        # Step 1: Candidate Generation
        candidates_per_step = []
        for pt in points:
            lat = pt.get("raw_latitude", pt.get("lat"))
            lon = pt.get("raw_longitude", pt.get("lon"))
            heading = pt.get("heading_deg", pt.get("heading"))
            speed = pt.get("speed_mps", pt.get("speed"))
            
            cands = self.candidate_matcher.get_candidates(
                lat=lat, lon=lon, heading=heading, speed_mps=speed, max_candidates=5
            )
            candidates_per_step.append(cands)

        # Step 2: HMM Viterbi Global Optimization
        matched_sequence = self.hmm_matcher.viterbi_decode(points, candidates_per_step)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 3: Result Formatting & Metrics
        results = []
        correct_count = 0
        total_eval_points = 0
        total_snapping_error_m = 0.0

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
            "avg_snapping_distance_m": avg_snapping_dist,
            "matched_edges_sequence": route_sequence,
            "snapped_points": results
        }

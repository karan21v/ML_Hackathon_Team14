"""
HMM Map-Matching with Viterbi Decoding.
Combines emission probabilities from road candidates and transition
probabilities from network shortest path routing.
"""

import math
import networkx as nx
import numpy as np
from src.models.candidate_matcher import haversine_distance

class HMMMapMatcher:
    def __init__(self, edges_df, sigma_z=12.0, beta=15.0):
        """
        sigma_z: Standard deviation of GPS noise in meters (emission parameter).
        beta: Parameter for exponential distribution of route distance discrepancy (transition parameter).
        """
        self.edges_df = edges_df.copy()
        self.sigma_z = sigma_z
        self.beta = beta
        self._build_road_graph()

    def _build_road_graph(self):
        """Build directed graph for shortest path routing between road segments."""
        self.graph = nx.DiGraph()
        self.edge_info = {}
        for _, row in self.edges_df.iterrows():
            u = str(row["from_node"])
            v = str(row["to_node"])
            eid = str(row["edge_id"])
            length = float(row["length_meters"])
            self.graph.add_edge(u, v, weight=length, edge_id=eid)
            self.edge_info[eid] = {
                "from_node": u,
                "to_node": v,
                "length": length,
                "u_lat": float(row["u_lat"]),
                "u_lon": float(row["u_lon"]),
                "v_lat": float(row["v_lat"]),
                "v_lon": float(row["v_lon"]),
            }

    def compute_route_distance(self, cand_prev, cand_curr):
        """
        Compute network distance from previous candidate projection to current candidate projection.
        """
        e_prev_id = cand_prev["edge_id"]
        e_curr_id = cand_curr["edge_id"]
        
        info_prev = self.edge_info.get(e_prev_id)
        info_curr = self.edge_info.get(e_curr_id)
        if not info_prev or not info_curr:
            return 9999.0

        # Case 1: Same edge
        if e_prev_id == e_curr_id:
            frac_diff = cand_curr["fraction_along"] - cand_prev["fraction_along"]
            if frac_diff >= 0:
                return frac_diff * info_prev["length"]
            else:
                # Driving backward on same directed edge: high penalty unless small noise jitter
                return abs(frac_diff) * info_prev["length"] * 3.0

        # Case 2: Different edges
        dist_to_end_of_prev = (1.0 - cand_prev["fraction_along"]) * info_prev["length"]
        dist_from_start_of_curr = cand_curr["fraction_along"] * info_curr["length"]

        node_a = info_prev["to_node"]
        node_b = info_curr["from_node"]

        if node_a == node_b:
            # Immediate direct connection
            return dist_to_end_of_prev + dist_from_start_of_curr

        # Graph shortest path between intersection nodes
        try:
            inter_dist = nx.shortest_path_length(self.graph, source=node_a, target=node_b, weight="weight")
            return dist_to_end_of_prev + inter_dist + dist_from_start_of_curr
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No topological path found in directed graph: apply heavy penalty
            return 9999.0

    def emission_log_prob(self, cand, point):
        """
        Log emission probability:
        p(z_t | c_i) = (1 / (sqrt(2*pi)*sigma_z)) * exp(-dist^2 / (2*sigma_z^2)) * ml_score
        """
        dist = cand["dist_to_segment_m"]
        gauss_term = -0.5 * (dist / self.sigma_z) ** 2
        norm_const = -math.log(math.sqrt(2 * math.pi) * self.sigma_z)
        ml_score = max(cand.get("ml_candidate_score", 1.0), 1e-4)
        ml_term = math.log(ml_score)
        return norm_const + gauss_term + ml_term

    def transition_log_prob(self, cand_prev, cand_curr, great_circle_dist):
        """
        Log transition probability:
        p(c_curr | c_prev) = (1 / beta) * exp(-|route_dist - great_circle_dist| / beta)
        """
        route_dist = self.compute_route_distance(cand_prev, cand_curr)
        dist_diff = abs(route_dist - great_circle_dist)
        log_prob = -math.log(self.beta) - (dist_diff / self.beta)
        return log_prob

    def viterbi_decode(self, trajectory_points, candidates_per_step):
        """
        Runs Viterbi dynamic programming to find optimal path across all time steps.
        trajectory_points: list of dicts with lat, lon
        candidates_per_step: list of lists of candidate dicts
        """
        T = len(trajectory_points)
        if T == 0:
            return []

        # Check for steps with empty candidates
        for t in range(T):
            if not candidates_per_step[t]:
                # Fallback: create dummy off-road candidate
                pt = trajectory_points[t]
                candidates_per_step[t] = [{
                    "edge_id": "OFF_ROAD",
                    "dist_to_segment_m": 100.0,
                    "proj_lat": pt["raw_latitude"],
                    "proj_lon": pt["raw_longitude"],
                    "fraction_along": 0.0,
                    "ml_candidate_score": 0.001
                }]

        # Trellis initialization
        trellis = []
        backpointers = []

        # Step 0
        v_0 = []
        for cand in candidates_per_step[0]:
            log_emit = self.emission_log_prob(cand, trajectory_points[0])
            v_0.append(log_emit)
        trellis.append(v_0)
        backpointers.append([None] * len(v_0))

        # Forward pass
        for t in range(1, T):
            pt_prev = trajectory_points[t - 1]
            pt_curr = trajectory_points[t]
            gcd = haversine_distance(
                pt_prev["raw_latitude"], pt_prev["raw_longitude"],
                pt_curr["raw_latitude"], pt_curr["raw_longitude"]
            )

            v_t = []
            bp_t = []

            for j, cand_curr in enumerate(candidates_per_step[t]):
                emit_log_p = self.emission_log_prob(cand_curr, pt_curr)
                best_score = -float("inf")
                best_prev_idx = 0

                for i, cand_prev in enumerate(candidates_per_step[t - 1]):
                    trans_log_p = self.transition_log_prob(cand_prev, cand_curr, gcd)
                    score = trellis[t - 1][i] + trans_log_p + emit_log_p
                    if score > best_score:
                        best_score = score
                        best_prev_idx = i

                v_t.append(best_score)
                bp_t.append(best_prev_idx)

            trellis.append(v_t)
            backpointers.append(bp_t)

        # Backtracking
        best_last_idx = int(np.argmax(trellis[-1]))
        matched_path = [None] * T
        curr_idx = best_last_idx

        for t in range(T - 1, -1, -1):
            matched_cand = dict(candidates_per_step[t][curr_idx])
            matched_cand["step_log_prob"] = float(trellis[t][curr_idx])
            matched_path[t] = matched_cand
            curr_idx = backpointers[t][curr_idx]

        return matched_path

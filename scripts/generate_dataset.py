"""
Data and Road Network Generator for PathSnap Hackathon.
Generates:
1. data/road_network/nodes.csv
2. data/road_network/edges.csv
3. data/raw/trips.csv (contains 5 realistic test trips with ground truth and noisy GPS)
4. data/raw/trip_<id>.json (individual trip files for evaluator interface)
"""

import json
import math
import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ROAD_DIR = os.path.join(DATA_DIR, "road_network")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ROAD_DIR, exist_ok=True)

# Bengaluru-like urban / arterial coordinates (around tech corridor / Koramangala / Indiranagar)
ORIGIN_LAT = 12.9352
ORIGIN_LON = 77.6245

def meters_to_latlon(dx, dy, ref_lat=ORIGIN_LAT):
    """Convert offset in meters to lat/lon degrees."""
    d_lat = dy / 111139.0
    d_lon = dx / (111139.0 * math.cos(math.radians(ref_lat)))
    return ref_lat + d_lat, ORIGIN_LON + d_lon

def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two lat/lon coordinates."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Define a realistic road grid with arterial roads, parallel avenues, and side streets
# Nodes in meters relative to ORIGIN
nodes_dict = {
    # Main Arterial Road (West to East)
    "N1": (0, 0),
    "N2": (500, 0),
    "N3": (1000, 0),
    "N4": (1500, 0),
    "N5": (2000, 0),
    
    # Parallel North Avenue (offset 60m North - tests parallel ambiguity)
    "N6": (0, 60),
    "N7": (500, 60),
    "N8": (1000, 60),
    "N9": (1500, 60),
    "N10": (2000, 60),
    
    # North Crossings
    "N11": (500, 500),
    "N12": (1000, 500),
    "N13": (1500, 500),
    
    # South Crossings / Tech Park
    "N14": (500, -500),
    "N15": (1000, -500),
    "N16": (1500, -500),
    
    # Ring Road Loop
    "N17": (0, 1000),
    "N18": (1000, 1000),
    "N19": (2000, 1000),
    "N20": (2000, -500),
}

# Convert nodes to real lat/lon
nodes_data = []
nodes_coords = {}
for nid, (x, y) in nodes_dict.items():
    lat, lon = meters_to_latlon(x, y)
    nodes_coords[nid] = (lat, lon)
    nodes_data.append({
        "node_id": nid,
        "x_meters": x,
        "y_meters": y,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6)
    })

nodes_df = pd.DataFrame(nodes_data)
nodes_df.to_csv(os.path.join(ROAD_DIR, "nodes.csv"), index=False)

# Edges (Road Segments)
edges_raw = [
    # Main Arterial (primary)
    ("E1", "N1", "N2", "primary", 50, 2),
    ("E2", "N2", "N3", "primary", 50, 2),
    ("E3", "N3", "N4", "primary", 50, 2),
    ("E4", "N4", "N5", "primary", 50, 2),
    
    # Parallel Avenue (residential / secondary)
    ("E5", "N6", "N7", "residential", 30, 1),
    ("E6", "N7", "N8", "residential", 30, 1),
    ("E7", "N8", "N9", "residential", 30, 1),
    ("E8", "N9", "N10", "residential", 30, 1),
    
    # Cross connectors
    ("E9", "N2", "N7", "secondary", 40, 1),
    ("E10", "N3", "N8", "secondary", 40, 1),
    ("E11", "N4", "N9", "secondary", 40, 1),
    
    # Northward Arterials
    ("E12", "N7", "N11", "secondary", 40, 1),
    ("E13", "N8", "N12", "secondary", 40, 1),
    ("E14", "N9", "N13", "secondary", 40, 1),
    
    # North Ring
    ("E15", "N11", "N12", "primary", 60, 2),
    ("E16", "N12", "N13", "primary", 60, 2),
    ("E17", "N17", "N18", "trunk", 80, 2),
    ("E18", "N18", "N19", "trunk", 80, 2),
    ("E19", "N12", "N18", "secondary", 40, 1),
    
    # South branch
    ("E20", "N2", "N14", "secondary", 40, 1),
    ("E21", "N3", "N15", "secondary", 40, 1),
    ("E22", "N4", "N16", "secondary", 40, 1),
    ("E23", "N14", "N15", "residential", 30, 1),
    ("E24", "N15", "N16", "residential", 30, 1),
    ("E25", "N16", "N20", "secondary", 50, 1),
    ("E26", "N5", "N20", "secondary", 50, 1),
]

# Create bidirected edges (two-way roads)
edges_data = []
for edge_id, u, v, road_type, speed_limit, lanes in edges_raw:
    u_lat, u_lon = nodes_coords[u]
    v_lat, v_lon = nodes_coords[v]
    length_m = haversine_distance(u_lat, u_lon, v_lat, v_lon)
    geom = [[u_lat, u_lon], [v_lat, v_lon]]
    
    # Forward direction
    edges_data.append({
        "edge_id": edge_id,
        "from_node": u,
        "to_node": v,
        "road_type": road_type,
        "speed_limit_kmh": speed_limit,
        "lanes": lanes,
        "length_meters": round(length_m, 2),
        "geometry": json.dumps(geom),
        "u_lat": u_lat, "u_lon": u_lon,
        "v_lat": v_lat, "v_lon": v_lon
    })
    
    # Reverse direction for map matching transitions
    edges_data.append({
        "edge_id": f"{edge_id}_rev",
        "from_node": v,
        "to_node": u,
        "road_type": road_type,
        "speed_limit_kmh": speed_limit,
        "lanes": lanes,
        "length_meters": round(length_m, 2),
        "geometry": json.dumps([[v_lat, v_lon], [u_lat, u_lon]]),
        "u_lat": v_lat, "u_lon": v_lon,
        "v_lat": u_lat, "v_lon": u_lon
    })

edges_df = pd.DataFrame(edges_data)
edges_df.to_csv(os.path.join(ROAD_DIR, "edges.csv"), index=False)

print(f"Generated {len(nodes_df)} nodes and {len(edges_df)} directed edges.")

# Trip trajectories definitions (sequences of connected edges)
trip_routes = [
    {
        "trip_id": "TRIP_001_MAIN_ARTERIAL",
        "description": "Straight high-speed trip along primary arterial corridor (E1 -> E2 -> E3 -> E4)",
        "edges": ["E1", "E2", "E3", "E4"],
        "base_speed_mps": 14.0,  # ~50 km/h
        "noise_sigma_m": 10.0,
        "sampling_dt_sec": 3.0
    },
    {
        "trip_id": "TRIP_002_PARALLEL_AMBIGUITY",
        "description": "Trip along parallel residential avenue (E5 -> E6 -> E7 -> E8) testing ambiguity against main road 60m away",
        "edges": ["E5", "E6", "E7", "E8"],
        "base_speed_mps": 8.0,   # ~30 km/h
        "noise_sigma_m": 14.0,
        "sampling_dt_sec": 2.5
    },
    {
        "trip_id": "TRIP_003_NORTH_TURNS",
        "description": "Trip with multiple 90-degree turns from arterial to north crossing (E1 -> E9 -> E12 -> E15 -> E16)",
        "edges": ["E1", "E9", "E12", "E15", "E16"],
        "base_speed_mps": 10.0,
        "noise_sigma_m": 12.0,
        "sampling_dt_sec": 2.0
    },
    {
        "trip_id": "TRIP_004_SOUTH_TECHPARK",
        "description": "Trip heading south into tech park corridor (E1 -> E2 -> E21 -> E24 -> E25)",
        "edges": ["E1", "E2", "E21", "E24", "E25"],
        "base_speed_mps": 9.5,
        "noise_sigma_m": 15.0,
        "sampling_dt_sec": 2.5
    },
    {
        "trip_id": "TRIP_005_HIGH_NOISE_URBAN",
        "description": "Challenging urban canyon trip with GPS multipath jitter and 22m noise (E6 -> E10 -> E3 -> E22 -> E24)",
        "edges": ["E6", "E10", "E3", "E22", "E24"],
        "base_speed_mps": 9.0,
        "noise_sigma_m": 22.0,
        "sampling_dt_sec": 3.0
    }
]

all_trip_points = []

for trip_spec in trip_routes:
    trip_id = trip_spec["trip_id"]
    edge_sequence = trip_spec["edges"]
    base_speed = trip_spec["base_speed_mps"]
    sigma_noise = trip_spec["noise_sigma_m"]
    dt = trip_spec["sampling_dt_sec"]
    
    trip_points = []
    timestamp = 1731916800  # 18-Nov-2026 09:30:00 UTC
    point_idx = 0
    
    for edge_id in edge_sequence:
        edge_row = edges_df[edges_df["edge_id"] == edge_id].iloc[0]
        u_lat, u_lon = edge_row["u_lat"], edge_row["u_lon"]
        v_lat, v_lon = edge_row["v_lat"], edge_row["v_lon"]
        length_m = edge_row["length_meters"]
        
        # Calculate heading in degrees
        d_lat = v_lat - u_lat
        d_lon = v_lon - u_lon
        heading_rad = math.atan2(d_lon * math.cos(math.radians(u_lat)), d_lat)
        heading_deg = (math.degrees(heading_rad) + 360) % 360
        
        # Number of samples along edge
        travel_time = length_m / base_speed
        n_samples = max(2, int(travel_time / dt))
        
        for s in range(n_samples):
            fraction = s / float(n_samples)
            # Ground truth location
            true_lat = u_lat + fraction * (v_lat - u_lat)
            true_lon = u_lon + fraction * (v_lon - u_lon)
            
            # Realistic GPS noise: Gaussian in X and Y meters
            noise_x = random.gauss(0, sigma_noise)
            noise_y = random.gauss(0, sigma_noise)
            # Occasional multipath spike for trip 5
            if trip_id == "TRIP_005_HIGH_NOISE_URBAN" and random.random() < 0.12:
                noise_x += random.choice([-1, 1]) * random.uniform(25, 45)
                noise_y += random.choice([-1, 1]) * random.uniform(25, 45)
                
            raw_lat, raw_lon = meters_to_latlon(noise_x, noise_y, ref_lat=true_lat)
            # Adjust lon according to true_lon
            raw_lon = true_lon + (raw_lon - ORIGIN_LON)
            
            # Speed with slight acceleration/deceleration noise
            speed = max(0.0, base_speed + random.gauss(0, 1.2))
            
            pt_record = {
                "trip_id": trip_id,
                "point_index": point_idx,
                "timestamp": timestamp,
                "raw_latitude": round(raw_lat, 6),
                "raw_longitude": round(raw_lon, 6),
                "speed_mps": round(speed, 2),
                "heading_deg": round(heading_deg, 1),
                "true_edge_id": edge_id,
                "true_latitude": round(true_lat, 6),
                "true_longitude": round(true_lon, 6),
                "noise_offset_meters": round(haversine_distance(raw_lat, raw_lon, true_lat, true_lon), 2)
            }
            
            trip_points.append(pt_record)
            all_trip_points.append(pt_record)
            point_idx += 1
            timestamp += int(dt + random.uniform(-0.4, 0.4))
            
    # Save individual trip JSON for evaluator UI
    trip_summary = {
        "trip_id": trip_id,
        "description": trip_spec["description"],
        "num_points": len(trip_points),
        "ground_truth_edges": edge_sequence,
        "noise_sigma_m": sigma_noise,
        "points": trip_points
    }
    with open(os.path.join(RAW_DIR, f"{trip_id}.json"), "w") as f:
        json.dump(trip_summary, f, indent=2)

trips_df = pd.DataFrame(all_trip_points)
trips_df.to_csv(os.path.join(RAW_DIR, "trips.csv"), index=False)
print(f"Generated {len(trip_routes)} trips with {len(trips_df)} total GPS points.")

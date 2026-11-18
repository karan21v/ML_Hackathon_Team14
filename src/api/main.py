"""
PathSnap FastAPI Server - Laptop 1 (Model and API).
Serves GPS trajectory map-matching predictions, evaluations,
network diagnostics, and embedded Evaluator Interface.
"""

import os
import sys
import json
import socket
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

# Path setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from src.models.pipeline import MapMatchingPipeline
from src.evaluation.map_visualizer import TrajectoryMapVisualizer
from src.api.schemas import (
    GPSPoint, SnappedPoint, MatchTripRequest,
    MatchTripResponse, TripSummary, HealthResponse
)

# Initialize FastAPI App
app = FastAPI(
    title="PathSnap API",
    description="Machine Learning & HMM-based GPS Trajectory Map-Matching Server for PathSnap Hackathon",
    version="1.0.0"
)

# Enable CORS for two-laptop network communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
ROAD_EDGES_PATH = os.path.join(DATA_DIR, "road_network", "edges.csv")
RAW_TRIPS_DIR = os.path.join(DATA_DIR, "raw")
UI_DIR = os.path.join(BASE_DIR, "ui")

# Load Road Network and Pipeline
if os.path.exists(ROAD_EDGES_PATH):
    edges_df = pd.read_csv(ROAD_EDGES_PATH)
else:
    edges_df = pd.DataFrame()

# Check for teammate trained model weights
teammate_model_path = os.path.join(BASE_DIR, "src", "models", "teammate_model.joblib")
if not os.path.exists(teammate_model_path):
    teammate_model_path = os.path.join(BASE_DIR, "src", "models", "model.joblib")

pipeline = MapMatchingPipeline(
    edges_df=edges_df,
    sigma_z=10.0,
    beta=12.0,
    ml_model_path=teammate_model_path if os.path.exists(teammate_model_path) else None
)
visualizer = TrajectoryMapVisualizer(edges_df=edges_df)

def get_local_ip_addresses():
    """Detect local IP addresses for two-laptop connectivity safely without DNS timeouts."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        if local_ip and not local_ip.startswith("127."):
            ips.append(local_ip)
        s.close()
    except Exception:
        pass

    if not ips:
        ips = ["127.0.0.1"]
    return ips

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"], response_model=HealthResponse)
def health():
    return {
        "status": "ONLINE",
        "server_time": datetime.now().isoformat(),
        "model_version": "PathSnap-HMM-Viterbi-v1.0",
        "active_engine": "ML-Candidate-Ranker + Viterbi-HMM",
        "loaded_edges": len(edges_df)
    }

@app.get("/api/network-status", tags=["Networking"])
def network_status():
    """Provides Laptop 1 IP addresses and instructions for Laptop 2 connection."""
    ips = get_local_ip_addresses()
    primary_ip = ips[0]
    return {
        "laptop_1_host": socket.gethostname(),
        "primary_ip": primary_ip,
        "all_detected_ips": ips,
        "port": 8000,
        "evaluator_ui_url": f"http://{primary_ip}:8000/ui",
        "api_docs_url": f"http://{primary_ip}:8000/docs",
        "instructions": (
            f"On Laptop 2 (Evaluator Interface), ensure connected to the same Wi-Fi / hotspot, "
            f"then open: http://{primary_ip}:8000/ui"
        )
    }

@app.get("/api/trips", tags=["Trips"], response_model=List[TripSummary])
def list_trips():
    """Lists all available test trips for the evaluator to pick from."""
    summaries = []
    if os.path.exists(RAW_TRIPS_DIR):
        for fname in sorted(os.listdir(RAW_TRIPS_DIR)):
            if fname.startswith("TRIP_") and fname.endswith(".json"):
                fpath = os.path.join(RAW_TRIPS_DIR, fname)
                try:
                    with open(fpath, "r") as f:
                        tdata = json.load(f)
                        summaries.append({
                            "trip_id": tdata.get("trip_id", fname.replace(".json", "")),
                            "description": tdata.get("description", "Evaluation trip"),
                            "num_points": tdata.get("num_points", len(tdata.get("points", []))),
                            "noise_sigma_m": tdata.get("noise_sigma_m", 0.0),
                            "ground_truth_edges": tdata.get("ground_truth_edges", [])
                        })
                except Exception:
                    continue
    return summaries

@app.get("/api/trips/{trip_id}", tags=["Trips"])
def get_trip(trip_id: str):
    """Retrieve full details and raw GPS points for a specific trip."""
    fpath = os.path.join(RAW_TRIPS_DIR, f"{trip_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
    with open(fpath, "r") as f:
        return json.load(f)

@app.post("/api/match", tags=["Map Matching"], response_model=MatchTripResponse)
def match_trajectory(payload: MatchTripRequest):
    """
    Core PathSnap Prediction Endpoint:
    Takes noisy GPS trajectory points, identifies candidates with ML model,
    and reconstructs the most likely road route with HMM Viterbi.
    """
    if not payload.points:
        raise HTTPException(status_code=400, detail="Empty points list provided")

    raw_pts_dict = [p.dict() for p in payload.points]
    
    # Configure custom HMM parameters if supplied
    pipeline.sigma_z = payload.sigma_z or 10.0
    pipeline.beta = payload.beta or 12.0
    pipeline.hmm_matcher.sigma_z = pipeline.sigma_z
    pipeline.hmm_matcher.beta = pipeline.beta

    result = pipeline.match_trip(raw_pts_dict)

    return {
        "trip_id": payload.trip_id or "SUBMITTED_TRIP",
        "num_points": result["num_points"],
        "latency_ms": result["latency_ms"],
        "accuracy": result["accuracy"],
        "avg_snapping_distance_m": result["avg_snapping_distance_m"],
        "matched_edges_sequence": result["matched_edges_sequence"],
        "snapped_points": result["snapped_points"]
    }

@app.post("/api/render-map", tags=["Visualization"])
def render_map(payload: MatchTripRequest):
    """Generates and returns an interactive Folium map as HTML."""
    raw_pts_dict = [p.dict() for p in payload.points]
    result = pipeline.match_trip(raw_pts_dict)

    # Extract ground truth edges if present
    gt_edges = []
    for p in raw_pts_dict:
        if p.get("true_edge_id"):
            e = p["true_edge_id"]
            if not gt_edges or gt_edges[-1] != e:
                gt_edges.append(e)

    folium_map = visualizer.create_folium_map(
        trip_id=payload.trip_id or "EVAL_TRIP",
        raw_points=raw_pts_dict,
        matched_results=result,
        ground_truth_edges=gt_edges or None
    )

    return HTMLResponse(content=folium_map.get_root().render(), status_code=200)

@app.get("/ui", response_class=HTMLResponse, tags=["Evaluator Interface"])
def serve_ui():
    """Serves the Evaluator Interface directly to Laptop 2 browser."""
    ui_html_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(ui_html_path):
        with open(ui_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Evaluator Interface Under Construction</h1>"

# Mount static directory for UI assets if present
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

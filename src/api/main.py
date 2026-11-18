"""
PathSnap FastAPI Server - Laptop 1 (Model and API).
Serves GPS trajectory map-matching predictions, evaluations,
network diagnostics, and embedded Evaluator Interface.
"""

import os
import sys
import json
import socket
import re
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
from src.api.real_trips import create_real_router
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

# Load the *real* training handoff when provided. Baseline mode is visibly
# distinguished in /health and in the evaluator dashboard.
configured_model_path = os.environ.get("PATHSNAP_MODEL_PATH")
model_paths = ([configured_model_path] if configured_model_path else [
    os.path.join(BASE_DIR, "artifacts", "handoff", "deploy_model.joblib"),
    os.path.join(BASE_DIR, "artifacts", "handoff", "best_model.joblib"),
    os.path.join(BASE_DIR, "src", "models", "teammate_model.joblib"),
    os.path.join(BASE_DIR, "src", "models", "model.joblib"),
])
if configured_model_path and not os.path.isfile(configured_model_path):
    raise RuntimeError(f"PATHSNAP_MODEL_PATH does not exist: {configured_model_path}")
selected_model_path = next((p for p in model_paths if p and os.path.isfile(p)), None)
if not selected_model_path and os.environ.get("PATHSNAP_REQUIRE_ML") == "1":
    raise RuntimeError("PATHSNAP_REQUIRE_ML=1 but trained model not found. Set PATHSNAP_MODEL_PATH.")
if not selected_model_path:
    print("[WARNING] No trained ML model found. Running clearly labelled GEOMETRIC BASELINE mode.")
pipeline = MapMatchingPipeline(
    edges_df=edges_df,
    sigma_z=10.0,
    beta=12.0,
    ml_model_path=selected_model_path,
)
visualizer = TrajectoryMapVisualizer(edges_df=edges_df)

# Real student-test trips are served from an explicitly configured ZIP.
# Their distinct road graphs must NEVER be mixed with synthetic demo edges.
app.include_router(create_real_router(lambda: pipeline.ml_model))

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
        "active_engine": pipeline.scoring_engine + " + Viterbi-HMM",
        "scoring_engine": pipeline.scoring_engine,
        "model_loaded": pipeline.ml_model is not None,
        "model_name": pipeline.model_name,
        "data_source": "synthetic_demo_trips" + (" + real_test_zip" if os.getenv("PATHSNAP_TRIPS_ZIP") else ""),
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
    """Retrieve synthetic demonstration trip data (not reserved evaluator trips)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trip_id):
        raise HTTPException(status_code=400, detail="Invalid trip ID")
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
    pipeline.sigma_z = payload.sigma_z if payload.sigma_z is not None else 10.0
    pipeline.beta = payload.beta if payload.beta is not None else 12.0
    pipeline.hmm_matcher.sigma_z = pipeline.sigma_z
    pipeline.hmm_matcher.beta = pipeline.beta

    try:
        result = pipeline.match_trip(raw_pts_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Map matching failed: {exc}") from exc

    return {
        "trip_id": payload.trip_id or "SUBMITTED_TRIP",
        "num_points": result["num_points"],
        "latency_ms": result["latency_ms"],
        "accuracy": result["accuracy"],
        "ml_only_accuracy": result["ml_only_accuracy"],
        "scoring_engine": result["scoring_engine"],
        "model_name": result["model_name"],
        "avg_snapping_distance_m": result["avg_snapping_distance_m"],
        "matched_edges_sequence": result["matched_edges_sequence"],
        "snapped_points": result["snapped_points"]
    }

@app.post("/api/match-with-map", tags=["Map Matching"])
def match_with_map(payload: MatchTripRequest):
    """One prediction and one map: avoids inconsistent latency/metrics."""
    if not payload.points:
        raise HTTPException(status_code=400, detail="Empty points list provided")
    if payload.sigma_z is not None and payload.sigma_z <= 0:
        raise HTTPException(status_code=422, detail="sigma_z must be positive")
    if payload.beta is not None and payload.beta <= 0:
        raise HTTPException(status_code=422, detail="beta must be positive")
    pipeline.hmm_matcher.sigma_z = payload.sigma_z if payload.sigma_z is not None else 10.0
    pipeline.hmm_matcher.beta = payload.beta if payload.beta is not None else 12.0
    raw = [p.model_dump() for p in payload.points]
    try:
        result = pipeline.match_trip(raw)
        gt = []
        for p in raw:
            edge_id = p.get("true_edge_id")
            if edge_id is not None and (not gt or gt[-1] != edge_id):
                gt.append(edge_id)
        fmap = visualizer.create_folium_map(
            trip_id=payload.trip_id or "DEMO_TRIP",
            raw_points=raw,
            matched_results=result,
            ground_truth_edges=gt or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Map matching failed: {exc}") from exc
    return {
        "prediction": {"trip_id": payload.trip_id or "CUSTOM_TRIP", **result},
        "map_html": fmap.get_root().render(),
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

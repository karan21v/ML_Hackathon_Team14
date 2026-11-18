"""
Pydantic Schemas for PathSnap FastAPI Server.
Defines clean data validation contracts for GPS points, trips,
matching requests, and evaluation results.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class GPSPoint(BaseModel):
    point_index: Optional[int] = None
    timestamp: Optional[float] = None
    raw_latitude: float = Field(..., alias="latitude", description="Raw WGS84 Latitude")
    raw_longitude: float = Field(..., alias="longitude", description="Raw WGS84 Longitude")
    speed_mps: Optional[float] = Field(None, alias="speed", description="Speed in m/s")
    heading_deg: Optional[float] = Field(None, alias="heading", description="Heading in degrees")
    true_edge_id: Optional[str] = Field(None, description="Ground truth road edge ID if available")

    class Config:
        populate_by_name = True

class SnappedPoint(BaseModel):
    point_index: int
    timestamp: Optional[float] = None
    raw_latitude: float
    raw_longitude: float
    snapped_latitude: float
    snapped_longitude: float
    matched_edge_id: str
    snapping_distance_m: float
    candidate_confidence: float
    true_edge_id: Optional[str] = None
    is_correct: Optional[bool] = None

class MatchTripRequest(BaseModel):
    trip_id: Optional[str] = "CUSTOM_TRIP"
    points: List[GPSPoint]
    sigma_z: Optional[float] = 12.0
    beta: Optional[float] = 15.0

class MatchTripResponse(BaseModel):
    trip_id: str
    num_points: int
    latency_ms: float
    accuracy: Optional[float] = None
    avg_snapping_distance_m: float
    matched_edges_sequence: List[str]
    snapped_points: List[SnappedPoint]

class TripSummary(BaseModel):
    trip_id: str
    description: str
    num_points: int
    noise_sigma_m: float
    ground_truth_edges: List[str]

class HealthResponse(BaseModel):
    status: str
    server_time: str
    model_version: str
    active_engine: str
    loaded_edges: int

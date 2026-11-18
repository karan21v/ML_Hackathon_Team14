"""Dependency-light tests: no FastAPI TestClient / httpx / httpx2 required."""
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from src.api.main import health, get_trip, match_with_map
from src.api.schemas import MatchTripRequest
from src.models.candidate_matcher import RoadCandidateMatcher


def test_health_exposes_actual_scoring_engine():
    info=health()
    assert "model_loaded" in info and "scoring_engine" in info
    assert info["loaded_edges"] > 0
    if not info["model_loaded"]:
        assert info["scoring_engine"] == "GEOMETRIC_BASELINE"


def test_map_and_metrics_are_same_inference():
    trip=get_trip("TRIP_001_MAIN_ARTERIAL")
    request=MatchTripRequest.model_validate({"trip_id":trip["trip_id"],
        "points":trip["points"],"sigma_z":10.0,"beta":12.0})
    response=match_with_map(request)
    pred=response["prediction"]
    assert pred["num_points"] == len(trip["points"])
    assert len(pred["snapped_points"]) == len(trip["points"])
    assert "PathSnap Matched Trajectory" in response["map_html"]
    assert pred["scoring_engine"] in ("GEOMETRIC_BASELINE","ML_MODEL")
    assert isinstance(pred["ml_only_accuracy"],float)


def test_missing_features_are_nan_and_scores_batched():
    class Probe:
        def __init__(self):self.calls=[]
        def predict_proba(self,m):
            m=np.asarray(m);self.calls.append(m)
            return np.tile([[0.4,0.6]],(len(m),1))
    edges=pd.read_csv('data/road_network/edges.csv')
    probe=Probe()
    matcher=RoadCandidateMatcher(edges,search_radius_m=65,ml_model={'model':probe})
    c=matcher.get_candidates(12.9352,77.629,heading=None,speed_mps=None,time_gap_s=None,max_candidates=5)
    assert c and len(probe.calls)==1
    assert probe.calls[0].shape==(len(c),8)
    assert np.isnan(probe.calls[0][:,1:4]).all()
    assert all(abs(row['ml_candidate_score']-0.6)<1e-8 for row in c)


def test_invalid_hmm_parameters_respond_422():
    trip=get_trip('TRIP_001_MAIN_ARTERIAL')
    request=MatchTripRequest.model_validate({'trip_id':trip['trip_id'],
          'points':trip['points'],'sigma_z':0,'beta':12})
    with pytest.raises(HTTPException) as e:
        match_with_map(request)
    assert e.value.status_code==422

"""Real PathSnap trip inference adapter.

This module reads only .track/.nodes/.arcs, computes the SAME eight features
as the official candidate handoff, scores candidates with the already-fitted
model, and decodes a directed-graph path by Viterbi. The .route files are
intentionally never opened. It is suitable for authorised testing, not for
training or hyperparameter optimisation.
"""
from __future__ import annotations

import io
import math
import os
import re
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import folium
import networkx as nx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pyproj import Transformer

from features.candidate_generation import (
    build_road_segments, generate_candidates, get_transformer,
)

FEATURES = (
    "distance_m", "heading_difference", "speed_mps", "time_gap_s",
    "segment_length_m", "candidate_rank", "projection_fraction", "heading_reliable"
)
TRIP_RE = re.compile(r"\d{8}\Z")


class RealMatchOptions(BaseModel):
    sigma_z: float = Field(default=8.0, gt=0, le=1000)
    beta: float = Field(default=12.0, gt=0, le=10000)
    window_start: int = Field(default=0, ge=0)
    max_points: int = Field(default=120, ge=0, le=20000)


class DatasetTripZip:
    """Supports Student-folder ZIP ('student_test') and explicitly authorised evaluator ZIP."""
    def __init__(self, path: str | Path, partition: str = "student_test"):
        self.path = Path(path)
        self.partition = partition
        if partition not in ("student_test", "evaluator_test"):
            raise ValueError("Only held-out test partitions are allowed")
        if partition == "evaluator_test" and os.getenv("PATHSNAP_ALLOW_EVALUATOR") != "1":
            raise ValueError("Evaluator set locked: set PATHSNAP_ALLOW_EVALUATOR=1 only when authorised")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.zip = zipfile.ZipFile(self.path)
        self.index = {}
        for name in self.zip.namelist():
            parts = name.strip("/").split("/")
            if len(parts) < 3 or parts[-3] != partition:
                continue
            trip_id, base = parts[-2:]
            if not TRIP_RE.fullmatch(trip_id) or "." not in base:
                continue
            stem, ext = base.rsplit(".", 1)
            if stem == trip_id and ext in ("track", "nodes", "arcs"):
                self.index[(trip_id, ext)] = name

    def trip_ids(self):
        return sorted(t for t in {k[0] for k in self.index}
                      if all((t, e) in self.index for e in ("track", "nodes", "arcs")))

    def read(self, trip_id: str, ext: str):
        if trip_id not in self.trip_ids() or ext not in ("track", "nodes", "arcs"):
            raise KeyError(f"Unknown trip or input type: {trip_id}.{ext}")
        raw = self.zip.read(self.index[(trip_id, ext)])
        dtype = int if ext == "arcs" else float
        values = np.loadtxt(io.BytesIO(raw), dtype=dtype, ndmin=2)
        expected = 3 if ext == "track" else 2
        if values.shape[1] != expected or values.shape[0] == 0:
            raise ValueError(f"Unexpected data shape for {ext}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid/nonfinite values in {ext}")
        return values

    def details(self, trip_id: str):
        if trip_id not in self.trip_ids():
            raise KeyError(trip_id)
        n = len(self.zip.read(self.index[(trip_id, "track")]).splitlines())
        return {"trip_id": trip_id, "num_points": n,
                "description": f"Real {self.partition.replace('_','-')} GPS trip; {n} observations. Reference route excluded from inference.",
                "dataset_type": self.partition, "is_real": True}

    def close(self):
        self.zip.close()


@dataclass
class RealTripMatcher:
    bundle: dict

    def __post_init__(self):
        if not isinstance(self.bundle, dict) or list(self.bundle.get("feature_names", [])) != list(FEATURES):
            raise ValueError("Saved model feature schema mismatch")
        if not hasattr(self.bundle.get("model"), "predict_proba"):
            raise ValueError("Saved model must expose predict_proba")

    def _load_inputs(self, dataset: DatasetTripZip, trip_id: str):
        track = dataset.read(trip_id, "track")
        nodes = dataset.read(trip_id, "nodes")
        arcs = dataset.read(trip_id, "arcs")
        if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
            raise ValueError("Arc references nonexistent road nodes")
        if np.any(np.diff(track[:,2]) < 0):
            raise ValueError("GPS timestamps are out of order")
        if np.any(np.abs(track[:,0]) > 180) or np.any(np.abs(track[:,1]) > 90):
            raise ValueError("Invalid GPS longitude/latitude")
        return track, nodes, arcs

    def match(self, dataset: DatasetTripZip, trip_id: str, opt: RealMatchOptions):
        tic = time.perf_counter()
        track, nodes, arcs = self._load_inputs(dataset, trip_id)
        total = len(track)
        start = opt.window_start
        if start >= total:
            raise ValueError(f"Window start {start} >= trip length {total}")
        end = total if opt.max_points == 0 else min(start + opt.max_points, total)
        if end - start < 2:
            raise ValueError("Need at least two GPS observations")
        window = track[start:end]
        transformer = get_transformer(float(np.mean(track[:,0])), float(np.mean(track[:,1])))
        segments = build_road_segments(nodes, arcs, transformer)
        cdf = generate_candidates(window, arcs, segments, transformer, k=5)
        if cdf.empty:
            raise ValueError("No road candidates produced")
        # Preserve original training-time GPS movement calculations; do not
        # infer speed from a stride or use synthetic constants for missing data.
        gps_x, gps_y = transformer.transform(track[:,0], track[:,1])
        dx, dy = np.diff(gps_x,prepend=np.nan), np.diff(gps_y,prepend=np.nan)
        dt = np.diff(track[:,2],prepend=np.nan)
        movement = np.hypot(dx,dy)
        speed = np.divide(movement,dt,out=np.full(total,np.nan),where=dt>0)
        gps_heading = np.degrees(np.arctan2(dx,dy)) % 360
        reliable = (movement>=5) & (dt>0) & (dt<=10)
        ax, ay, bx, by, mids, half_lengths = segments
        lengths = np.hypot(bx-ax,by-ay)
        road_heading = np.degrees(np.arctan2(bx-ax,by-ay)) % 360
        obs_local = cdf["observation_index"].to_numpy(dtype=int)
        orig_obs = obs_local+start
        ids = cdf["arc_id"].to_numpy(dtype=int)
        heading_diff = np.abs((gps_heading[orig_obs]-road_heading[ids]+180)%360-180)
        heading_diff[~reliable[orig_obs]] = np.nan
        X = np.column_stack([
            cdf["distance_m"].to_numpy(dtype=float),
            heading_diff, speed[orig_obs],dt[orig_obs],lengths[ids],
            cdf["candidate_rank"].to_numpy(dtype=float),
            cdf["projection_fraction"].to_numpy(dtype=float),
            reliable[orig_obs].astype(int)
        ])
        X[~np.isfinite(X)] = np.nan
        scores = np.asarray(self.bundle["model"].predict_proba(X),dtype=float)[:,1]
        if len(scores) != len(X) or not np.isfinite(scores).all():
            raise ValueError("Bad model probability output")
        cdf["ml_score"] = scores
        # Build directed road graph. Each arc ID is its row number in .arcs;
        # these are *not* the same IDs as the synthetic TRIP_* road network.
        graph = nx.DiGraph()
        for edge_id,(u,v) in enumerate(arcs):
            u,v = int(u),int(v)
            cost = float(lengths[edge_id])
            if graph.has_edge(u,v):
                graph[u][v]["weight"] = min(graph[u][v]["weight"],cost)
            else:
                graph.add_edge(u,v,weight=cost)
        per_step = []
        for local_idx in range(len(window)):
            grp = cdf[cdf["observation_index"] == local_idx]
            recs = []
            for r in grp.itertuples(index=False):
                arcid=int(r.arc_id)
                recs.append({"edge_id":str(arcid),"arc_id":arcid,
                             "start_node":int(arcs[arcid,0]),"end_node":int(arcs[arcid,1]),
                             "fraction":float(r.projection_fraction),"length":float(lengths[arcid]),
                             "distance":float(r.distance_m),"px":float(r.projected_x),"py":float(r.projected_y),
                             "score":float(r.ml_score)})
            if not recs:
                raise ValueError(f"No candidates for observation {local_idx}")
            per_step.append(recs)
        decoded = self._viterbi(window,per_step,graph,opt.sigma_z,opt.beta,transformer)
        inverse = Transformer.from_crs(transformer.target_crs,"EPSG:4326",always_xy=True)
        points=[]; route=[]
        for i,c in enumerate(decoded):
            lon,lat=inverse.transform(c["px"],c["py"])
            row={"point_index":int(i+start),"timestamp":float(window[i,2]),
                 "raw_latitude":float(window[i,1]),"raw_longitude":float(window[i,0]),
                 "snapped_latitude":float(lat),"snapped_longitude":float(lon),
                 "matched_edge_id":c["edge_id"], "snapping_distance_m":round(c["distance"],2),
                 "candidate_confidence":round(c["score"],5)}
            points.append(row)
            if not route or route[-1] != c["edge_id"]:
                route.append(c["edge_id"])
        elapsed = round((time.perf_counter()-tic)*1000,1)
        prediction={"trip_id":trip_id,"num_points":len(window),"total_trip_points":total,
                    "window_start":start,"preview_only":len(window)!=total,
                    "accuracy":None,"ml_only_accuracy":None,"latency_ms":elapsed,
                    "avg_snapping_distance_m":round(float(np.mean([p["snapping_distance_m"] for p in points])),2),
                    "scoring_engine":"ML_MODEL","model_name":self.bundle.get("model_name","unknown"),
                    "matched_edges_sequence":route,"snapped_points":points,
                    "notes":"Real trip with no reference-route labels read. Accuracy unavailable. A preview uses only consecutive GPS observations."}
        fmap=self._map(trip_id,points,arcs,nodes)
        return {"prediction":prediction,"map_html":fmap.get_root().render()}

    @staticmethod
    def _viterbi(track, candidates, graph, sigma, beta, transformer):
        # Bounded directed Dijkstra for responsive demonstrations. For every
        # transition the bound is adaptive to the observed GPS displacement.
        # If no directed route is found inside the bound, penalise heavily.
        score_prev=[-0.5*(c["distance"]/sigma)**2+math.log(max(c["score"],1e-6)) for c in candidates[0]]
        backs=[None]
        for t in range(1,len(candidates)):
            p=track[t-1];q=track[t]
            px,py=transformer.transform(float(p[0]),float(p[1]))
            qx,qy=transformer.transform(float(q[0]),float(q[1]))
            observed_dist=math.hypot(qx-px,qy-py)
            cutoff=max(1000.,3*observed_dist+300.)
            reachable={}
            for prev in candidates[t-1]:
                src=prev["end_node"]
                if src not in reachable:
                    reachable[src]=nx.single_source_dijkstra_path_length(graph,src,cutoff=cutoff,weight="weight")
            score_next=[]; back=[]
            for nxt in candidates[t]:
                emit=-0.5*(nxt["distance"]/sigma)**2+math.log(max(nxt["score"],1e-6))
                best=-float("inf"); best_idx=0
                for i,prev in enumerate(candidates[t-1]):
                    if prev["arc_id"]==nxt["arc_id"]:
                        df=nxt["fraction"]-prev["fraction"]
                        route_dist=df*prev["length"] if df>=0 else -3*df*prev["length"]
                    else:
                        mid=reachable[prev["end_node"]].get(nxt["start_node"])
                        route_dist=(1-prev["fraction"])*prev["length"]+mid+nxt["fraction"]*nxt["length"] if mid is not None else 9999.
                    val=score_prev[i]+emit-abs(route_dist-observed_dist)/beta
                    if val>best:best=val;best_idx=i
                score_next.append(best);back.append(best_idx)
            backs.append(back);score_prev=score_next
        idx=int(np.argmax(score_prev)); chosen=[]
        for t in range(len(candidates)-1,-1,-1):
            chosen.append(candidates[t][idx]);
            if t:idx=backs[t][idx]
        return list(reversed(chosen))

    @staticmethod
    def _map(trip_id, points, arcs, nodes):
        c=[(p["raw_latitude"],p["raw_longitude"]) for p in points]
        s=[(p["snapped_latitude"],p["snapped_longitude"]) for p in points]
        fmap=folium.Map(location=c[len(c)//2],zoom_start=14,tiles="OpenStreetMap",control_scale=True)
        raw=folium.FeatureGroup(name="Raw GPS fixes",show=True)
        folium.PolyLine(c,color="#f59e0b",weight=2,opacity=.6).add_to(raw)
        for lat,lon in c[::max(1,len(c)//100)]:
            folium.CircleMarker([lat,lon],radius=2,tooltip="Raw GPS",color="#f59e0b").add_to(raw)
        raw.add_to(fmap)
        matched=folium.FeatureGroup(name="Viterbi matched points (connected for display)",show=True)
        folium.PolyLine(s,color="#1565c0",weight=4,opacity=.85).add_to(matched)
        folium.Marker(s[0],tooltip="Start",icon=folium.Icon(color="green")).add_to(matched)
        folium.Marker(s[-1],tooltip="End",icon=folium.Icon(color="red")).add_to(matched)
        matched.add_to(fmap)
        edges=folium.FeatureGroup(name="Selected candidate road arcs",show=True)
        for arc_id in dict.fromkeys(int(p["matched_edge_id"]) for p in points):
            u,v=arcs[arc_id]
            folium.PolyLine([[float(nodes[u,1]),float(nodes[u,0])],
                             [float(nodes[v,1]),float(nodes[v,0])]],
                            color="#2e7d32",weight=3,opacity=.65,tooltip=f"Arc {arc_id}").add_to(edges)
        edges.add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.fit_bounds(c)
        return fmap


def create_real_router(get_bundle: Callable[[], Optional[dict]]):
    """Optional real-trip endpoints. No evaluator data is read unless explicitly configured."""
    router=APIRouter(prefix="/api/real",tags=["Real Test Trips"])
    @lru_cache(maxsize=1)
    def dataset():
        path=os.getenv("PATHSNAP_TRIPS_ZIP")
        if not path:
            return None
        return DatasetTripZip(path,os.getenv("PATHSNAP_TRIP_PARTITION","student_test"))
    def ensure():
        try:
            ds=dataset()
        except Exception as exc:
            raise HTTPException(status_code=503,detail=str(exc)) from exc
        if ds is None:
            raise HTTPException(status_code=503,detail="Set PATHSNAP_TRIPS_ZIP to enable real trips")
        return ds
    @router.get("/trips")
    def trips():
        ds=ensure()
        return [ds.details(t) for t in ds.trip_ids()]
    @router.get("/trips/{trip_id}")
    def detail(trip_id:str):
        ds=ensure()
        try:return ds.details(trip_id)
        except KeyError:raise HTTPException(status_code=404,detail="Trip not found")
    @router.post("/match/{trip_id}")
    def match(trip_id:str,opt:RealMatchOptions):
        ds=ensure()
        if trip_id not in ds.trip_ids():
            raise HTTPException(status_code=404,detail="Trip not found")
        bundle=get_bundle()
        if bundle is None:
            raise HTTPException(status_code=503,detail="Real trips require a loaded trained model; baseline disabled")
        try:return RealTripMatcher(bundle).match(ds,trip_id,opt)
        except (ValueError,IndexError) as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
    return router

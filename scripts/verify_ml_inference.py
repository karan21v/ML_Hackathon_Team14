#!/usr/bin/env python3
"""PathSnap inference-contract smoke test on one trip (no labels used).

Exports candidate probabilities for HMM integration. This is NOT an unbiased
accuracy evaluation when the input comes from the training CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_one_trip(csv_path: Path, requested_trip: str | None) -> tuple[pd.DataFrame, str]:
    wanted = requested_trip.zfill(8) if requested_trip else None
    chunks = []
    chosen = wanted
    for block in pd.read_csv(csv_path, dtype={"trip_id": str}, chunksize=100_000, low_memory=False):
        if "trip_id" not in block:
            raise ValueError("CSV requires trip_id column")
        block["trip_id"] = block["trip_id"].astype(str).str.zfill(8)
        if chosen is None:
            if block.empty:
                continue
            chosen = str(block["trip_id"].iloc[0])
        picked = block.loc[block["trip_id"] == chosen]
        if not picked.empty:
            chunks.append(picked.copy())
    if not chunks:
        raise ValueError(f"Trip {requested_trip!r} not found in {csv_path}")
    return pd.concat(chunks, ignore_index=True), str(chosen)


def score_trip(bundle: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required_bundle = {"model", "feature_names", "model_name"}
    if not isinstance(bundle, dict) or not required_bundle <= set(bundle):
        raise ValueError("Expected a PathSnap deployment bundle with model and feature_names")
    features = list(bundle["feature_names"])
    if len(features) != 8 or len(set(features)) != 8:
        raise ValueError(f"Expected eight distinct features; got: {features}")
    if "obs_idx" in df and "observation_index" not in df:
        df = df.rename(columns={"obs_idx": "observation_index"})
    needed = ["trip_id", "observation_index", "arc_id", *features]
    missing = [c for c in needed if c not in df]
    if missing:
        raise ValueError(f"Missing inference fields: {missing}")
    if df["observation_index"].isna().any() or df["arc_id"].isna().any():
        raise ValueError("Null observation_index/arc_id in inference input")
    if df.duplicated(["trip_id", "observation_index", "arc_id"]).any():
        raise ValueError("Duplicate arc_id within an observation")

    numeric = df[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    missing_counts = numeric.isna().sum().to_dict()
    # The trained SimpleImputer belongs to the saved Pipeline. Do not manually impute.
    X = numeric.to_numpy(dtype=np.float32)
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not expose predict_proba for HMM emissions")
    t0 = time.perf_counter()
    probs = np.asarray(model.predict_proba(X))
    elapsed = time.perf_counter() - t0
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Expected class 1 in model classes_; got {classes}")
    if probs.ndim != 2 or probs.shape[0] != len(df):
        raise ValueError(f"Invalid probability output shape: {probs.shape}")
    scores = np.asarray(probs[:, classes.index(1)], dtype=np.float64)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Nonfinite or out-of-range positive-class probabilities")

    # No label or reference-route columns are sent to inference or exported.
    scored = df[["trip_id", "observation_index", "arc_id"]].copy()
    scored["candidate_score"] = scores
    scored["emission_log_score"] = np.log(np.clip(scores, 1e-9, 1.0))
    scored = scored.sort_values(["observation_index", "candidate_score"], ascending=[True, False], kind="stable")
    grouped = scored.groupby(["trip_id", "observation_index"], sort=False)
    winner = grouped.head(1).copy().rename(columns={"arc_id": "predicted_arc_id", "candidate_score": "top1_score"})
    winner = winner.drop(columns=["emission_log_score"])
    counts = grouped.size()
    runner_up = grouped["candidate_score"].nth(1).reset_index(drop=True)
    if len(runner_up) == len(winner):
        winner["runner_up_score"] = runner_up.to_numpy()
        winner["top1_margin"] = winner["top1_score"] - winner["runner_up_score"]
    else:
        winner["runner_up_score"] = np.nan
        winner["top1_margin"] = np.nan

    report = {
        "purpose": "Unseen-input schema/probability contract smoke test; not independent accuracy evidence",
        "model": bundle["model_name"],
        "experiment_id": bundle.get("experiment_id"),
        "feature_order": features,
        "trip_id": str(scored["trip_id"].iloc[0]),
        "candidate_rows": int(len(scored)),
        "gps_observations": int(len(winner)),
        "candidates_per_observation": {
            "min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max())
        },
        "missing_features": {k: int(v) for k, v in missing_counts.items()},
        "positive_class_score_range": [float(scores.min()), float(scores.max())],
        "inference_seconds": round(elapsed, 4),
        "warning": "Scores are independent candidate class probabilities; not normalized to sum to 1 within observations. HMM must handle transition scores itself.",
    }
    return scored, winner, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="artifacts/handoff/deploy_model.joblib")
    p.add_argument("--csv", default="data/handoff/candidate_features_handoff.csv")
    p.add_argument("--trip", default=None, help="trip_id; default first trip found")
    p.add_argument("--out", default="results/ml_inference_audit")
    args = p.parse_args()
    model_path, csv_path = Path(args.model), Path(args.csv)
    if not model_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("Missing model or CSV; run from the project root")
    bundle = joblib.load(model_path)  # Only load your team's trusted local artifact.
    df, tid = load_one_trip(csv_path, args.trip)
    scored, winners, report = score_trip(bundle, df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scored_path = out / f"candidate_scores_{tid}.csv"
    winner_path = out / f"top1_candidates_{tid}.csv"
    report_path = out / f"inference_audit_{tid}.json"
    scored.to_csv(scored_path, index=False)
    winners.to_csv(winner_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("[OK] Model:", report["model"], "| trip:", tid)
    print("[OK] Rows:", report["candidate_rows"], "| observations:", report["gps_observations"])
    print("[OK] Candidate probability validation passed")
    print("[SAVED]", scored_path)
    print("[SAVED]", winner_path)
    print("[SAVED]", report_path)
    print("[NOTE] These are model scores, NOT HMM-decoded arcs or held-out accuracy.")


if __name__ == "__main__":
    main()

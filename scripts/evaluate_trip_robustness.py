#!/usr/bin/env python3
"""Trip-wise out-of-fold Random Forest robustness on the handoff training CSV.

Uses only fold-held-out predictions, never in-sample scores, and keeps a trip
wholly in either training or validation. Labels remain geometry-derived proxy
labels, not independently verified true road segments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pathsnap.modeling import model_factory
from train_handoff_csv import load_table, with_fold_safe_imputation


def per_trip_metrics(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Compute descriptive proxy metrics from OOF probabilities only."""
    p = np.asarray(proba, dtype=np.float64)
    if len(p) != len(df) or not np.isfinite(p).all():
        raise ValueError("OOF predictions are missing or invalid")
    tmp = df[["trip_id", "observation_index", "label"]].copy()
    tmp["score"] = p
    tmp["pred"] = (p >= 0.5).astype(np.int8)
    # Take the highest-score road candidate for each GPS observation.
    top1idx = tmp.groupby(["trip_id", "observation_index"], sort=False)["score"].idxmax()
    correct = tmp.loc[top1idx, ["trip_id", "label"]].groupby("trip_id")["label"].agg(["mean", "size"])
    correct.columns = ["observation_top1_proxy", "observations"]
    records = []
    for tid, g in tmp.groupby("trip_id", sort=True):
        y = g["label"].to_numpy(dtype=np.int8)
        prob = np.clip(g["score"].to_numpy(), 1e-7, 1 - 1e-7)
        records.append({
            "trip_id": str(tid),
            "observations": int(correct.loc[tid, "observations"]),
            "candidate_rows": int(len(g)),
            "candidate_macro_f1_proxy": float(f1_score(y, g["pred"], average="macro", zero_division=0)),
            "observation_top1_proxy": float(correct.loc[tid, "observation_top1_proxy"]),
            "candidate_log_loss_proxy": float(log_loss(y, prob, labels=[0, 1])),
        })
    return pd.DataFrame(records).sort_values("observation_top1_proxy").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="data/handoff/candidate_features_handoff.csv")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--model", default="random_forest", choices=sorted(model_factory()))
    p.add_argument("--max-obs-per-trip", type=int, default=0)
    p.add_argument("--out", default="results/trip_robustness")
    args = p.parse_args()

    if args.folds < 2:
        raise ValueError("Use at least two GroupKFold folds")
    df, features, stats = load_table(Path(args.csv), args.max_obs_per_trip)
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int8)
    groups = df["trip_id"].astype(str).to_numpy()
    if len(set(groups)) < args.folds:
        raise ValueError("Not enough trips for requested folds")
    pred = np.full(len(df), np.nan, dtype=np.float64)
    for n, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups), 1):
        tr_tids = set(groups[train_idx]); va_tids = set(groups[valid_idx])
        if tr_tids & va_tids:
            raise RuntimeError("Data leakage: same trip in train and validation")
        print(f"[FOLD {n}/{args.folds}] training on {len(train_idx):,} rows, validating {len(valid_idx):,} rows", flush=True)
        model = with_fold_safe_imputation(clone(model_factory()[args.model]))
        model.fit(X[train_idx], y[train_idx])
        pred[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]
    metrics = per_trip_metrics(df, pred)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "per_trip_oof_metrics.csv", index=False)
    vals = metrics["observation_top1_proxy"].to_numpy(dtype=float)
    summary = {
        "model": args.model, "folds": args.folds,
        "candidate_rows": int(len(df)), "trips": int(df.trip_id.nunique()),
        "observations": int(stats["training_observations"]),
        "mean_trip_top1_proxy_unweighted": float(np.mean(vals)),
        "median_trip_top1_proxy": float(np.median(vals)),
        "p10_trip_top1_proxy": float(np.quantile(vals, 0.10)),
        "min_trip_top1_proxy": float(np.min(vals)),
        "max_trip_top1_proxy": float(np.max(vals)),
        "warning": "Out-of-fold TRAINING-trips with approximate reference-alignment labels, not the final reserved evaluator set. Do not interpret as HMM route accuracy.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals, bins=min(18, max(5, len(vals)//4)))
    ax.set_title(f"{args.model}: OOF observation Top-1 proxy by trip ({len(vals)} trips)")
    ax.set_xlabel("Trip-level Top-1 proxy accuracy")
    ax.set_ylabel("Number of trips")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=.25)
    fig.text(.01, .01, "PathSnap Team 14 | Trip-grouped out-of-fold | Proxy labels", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out / "trip_top1_distribution.png", dpi=180)
    plt.close(fig)
    print("[SAVED]", out / "per_trip_oof_metrics.csv")
    print("[SAVED]", out / "summary.json")
    print("[SAVED]", out / "trip_top1_distribution.png")
    print("[SUMMARY]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

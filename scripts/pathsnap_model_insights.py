#!/usr/bin/env python3
"""PathSnap ML explainability + candidate ambiguity audit.

Uses the final fitted Random Forest and existing candidate-score CSV exported by
scripts/verify_ml_inference.py. No retraining and no reference labels.

IMPORTANT: Scores are independent candidate-class probabilities, not calibrated
road-choice probabilities. Ambiguity flags use a heuristic, not a learned
uncertainty threshold; HMM should still receive every candidate score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_importances(bundle: dict) -> pd.DataFrame:
    """Return importance for *transformed* model inputs, including missing flags.

    Training uses SimpleImputer(add_indicator=True). The raw feature count is 8,
    but the fitted estimator may see extra binary missingness indicators. Using
    get_feature_names_out prevents incorrectly pairing 11 importances with 8 names.
    """
    features = bundle.get('feature_names')
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError('Model bundle does not specify feature_names')
    model = bundle.get('model')
    if model is None:
        raise ValueError('Model bundle lacks fitted model')
    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]
        preprocessors = model.steps[:-1]
        names = np.asarray(features, dtype=str)
        for _, transformer in preprocessors:
            if not hasattr(transformer, 'get_feature_names_out'):
                raise TypeError('Preprocessing step lacks feature-name mapping; cannot label importances safely')
            names = np.asarray(transformer.get_feature_names_out(names), dtype=str)
    else:
        estimator = model
        names = np.asarray(features, dtype=str)
    if not hasattr(estimator, 'feature_importances_'):
        raise TypeError('Selected model does not support tree feature importances')
    importance = np.asarray(estimator.feature_importances_, dtype=float)
    if importance.ndim != 1 or len(importance) != len(names) or not np.isfinite(importance).all():
        raise ValueError(f'Unexpected importance dimensions: model={importance.shape}, transformed names={len(names)}')
    if np.any(importance < 0):
        raise ValueError('Negative feature importance')
    return pd.DataFrame({'feature': names, 'importance': importance}).sort_values('importance', ascending=False)


def analyze_candidate_scores(scored: pd.DataFrame, margin_threshold: float, top_score_threshold: float):
    need = {'trip_id', 'observation_index', 'arc_id', 'candidate_score'}
    missing = need - set(scored.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if scored.empty:
        raise ValueError('Empty scored CSV')
    data = scored.loc[:, list(need)].copy()
    data['trip_id'] = data['trip_id'].astype(str).str.zfill(8)
    data['candidate_score'] = pd.to_numeric(data['candidate_score'], errors='raise')
    if data[['trip_id','observation_index','arc_id','candidate_score']].isna().any().any():
        raise ValueError('Null identifier/score found')
    if not np.isfinite(data['candidate_score'].to_numpy(dtype=float)).all():
        raise ValueError('NaN or infinite candidate score')
    if not data['candidate_score'].between(0, 1).all():
        raise ValueError('Scores must be in [0,1]')
    if data.duplicated(['trip_id', 'observation_index', 'arc_id']).any():
        raise ValueError('Duplicate candidate arc within observation')
    data = data.sort_values(['trip_id', 'observation_index', 'candidate_score', 'arc_id'], ascending=[True, True, False, True], kind='stable')
    rows = []
    for (tid, obs_idx), block in data.groupby(['trip_id','observation_index'], sort=False):
        best = block.iloc[0]
        second = block.iloc[1] if len(block)>1 else None
        second_score = float(second.candidate_score) if second is not None else np.nan
        margin = float(best.candidate_score) - second_score if second is not None else np.nan
        # Heuristic ambiguity only, NOT evidence of incorrect route or calibration.
        flag = (second is not None and margin <= margin_threshold) or (float(best.candidate_score) < top_score_threshold)
        rows.append({
            'trip_id': tid,
            'observation_index': obs_idx,
            'num_candidates': len(block),
            'top1_arc_id': best.arc_id,
            'top1_score': float(best.candidate_score),
            'top2_arc_id': second.arc_id if second is not None else None,
            'top2_score': second_score,
            'top1_top2_margin': margin,
            'ambiguous_heuristic': bool(flag),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='artifacts/handoff/deploy_model.joblib')
    p.add_argument('--scores', default='results/ml_inference_audit/candidate_scores_00000001.csv')
    p.add_argument('--output', default='results/model_insights')
    p.add_argument('--margin-threshold', type=float, default=0.10)
    p.add_argument('--top-score-threshold', type=float, default=0.60)
    args = p.parse_args()
    if not 0 <= args.margin_threshold <= 1 or not 0 <= args.top_score_threshold <= 1:
        p.error('Thresholds must be in [0,1]')
    if not Path(args.model).is_file() or not Path(args.scores).is_file():
        p.error('Model or score CSV missing; run verify_ml_inference.py before this script')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Only load model files created by your own trusted training pipeline.
    bundle = joblib.load(args.model)
    importance = get_importances(bundle)
    importance.to_csv(out/'feature_importance.csv', index=False)
    fig, ax = plt.subplots(figsize=(8.5,4.7))
    ordered = importance.iloc[::-1]
    ax.barh(ordered.feature, ordered.importance)
    ax.set_xlabel('Random Forest impurity-based importance (not causal)')
    ax.set_title('PathSnap: learned feature importance')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'feature_importance.png',dpi=180)
    plt.close(fig)

    scored = pd.read_csv(args.scores, dtype={'trip_id':str, 'arc_id':str})
    ranked = analyze_candidate_scores(scored,args.margin_threshold,args.top_score_threshold)
    ranked.to_csv(out/'candidate_ambiguity.csv',index=False)
    margins = ranked['top1_top2_margin'].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5,4.7))
    if len(margins):
        ax.hist(margins,bins=25)
    ax.axvline(args.margin_threshold,linestyle='--',label='Ambiguity margin threshold')
    ax.set_xlabel('Top-1 minus Top-2 candidate score')
    ax.set_ylabel('GPS observation count')
    ax.set_title('PathSnap: road candidate score ambiguity (not accuracy)')
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'candidate_ambiguity.png',dpi=180)
    plt.close(fig)

    summary = {
        'model': bundle.get('model_name','unknown'),
        'trip_count': int(ranked.trip_id.nunique()),
        'gps_observations': int(len(ranked)),
        'candidate_rows': int(len(scored)),
        'ambiguous_observations': int(ranked.ambiguous_heuristic.sum()),
        'ambiguous_fraction': float(ranked.ambiguous_heuristic.mean()),
        'margin_threshold': args.margin_threshold,
        'top_score_threshold': args.top_score_threshold,
        'top_features': importance.head(3).to_dict(orient='records'),
        'note': 'Descriptive analysis of model scores for input trip(s), not unseen-trip accuracy. Thresholds are heuristic; candidate scores are not calibrated across candidates; HMM still needs all candidates.'
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('[OK] Saved:',out/'feature_importance.csv',out/'feature_importance.png')
    print('[OK] Saved:',out/'candidate_ambiguity.csv',out/'candidate_ambiguity.png')
    print('[SUMMARY]',json.dumps(summary,indent=2))

if __name__=='__main__':
    main()

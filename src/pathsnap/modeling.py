"""Reproducible model factory and strictly trip-grouped comparison metrics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score

SEED=20261118


def model_factory():
    return {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=350, class_weight='balanced', random_state=SEED)),
        'decision_tree': DecisionTreeClassifier(max_depth=15, min_samples_leaf=8, random_state=SEED, class_weight='balanced'),
        'random_forest': RandomForestClassifier(n_estimators=65, min_samples_leaf=3, n_jobs=-1, random_state=SEED, class_weight='balanced_subsample'),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
            solver='adam', alpha=0.003, learning_rate_init=0.001,
            max_iter=100, early_stopping=True, n_iter_no_change=8,
            random_state=SEED)),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=25, learning_rate=0.08, random_state=SEED),
    }


def metrics_for_candidates(y, proba, trip_ids, observation_ids):
    """Point-level correctness counts candidate misses as incorrect.

    y uses geometry-derived proxy reference labels, not exact observation GT.
    """
    proba=np.asarray(proba, dtype=float)
    pred=(proba>=0.5).astype(int)
    # Group one predicted candidate per (trip, observation), where rows are already
    # sorted by trip and observation after feature creation.
    choices={}
    hit=set()
    for i,(tid,oid) in enumerate(zip(trip_ids,observation_ids)):
        key=(str(tid),int(oid))
        if y[i]==1:hit.add(key)
        if key not in choices or proba[i]>choices[key][0]: choices[key]=(proba[i], bool(y[i]))
    return {
        'candidate_macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'candidate_precision':float(precision_score(y,pred,zero_division=0)),
        'candidate_recall':float(recall_score(y,pred,zero_division=0)),
        'candidate_accuracy':float(accuracy_score(y,pred)),
        'candidate_log_loss':float(log_loss(y,np.clip(proba,1e-7,1-1e-7),labels=[0,1])),
        'observation_top1_accuracy_proxy': float(sum(v[1] for v in choices.values())/max(len(choices),1)),
        'candidate_miss_rate_proxy':float(1-len(hit)/max(len(choices),1)),
        'observations':int(len(choices)),
    }

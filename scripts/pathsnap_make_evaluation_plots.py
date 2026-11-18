#!/usr/bin/env python3
"""PathSnap Team 14: reproducible evaluation plots from saved results / grouped-CV predictions.
Run from the repository root. For a full confusion-matrix evaluation use --confusion.
This program does not use evaluator-reserved trips or invented predictions.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DISPLAY = {
    'random_forest': 'Random Forest',
    'hist_gradient_boosting': 'HistGradientBoosting',
    'mlp': 'Neural Network (MLP)',
    'decision_tree': 'Decision Tree',
    'logistic_regression': 'Logistic Regression',
}

def safe_name(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name))

def save(fig, out, name):
    f = out / name
    fig.savefig(f, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[SAVED]', f)

def named(df):
    return [DISPLAY.get(x, str(x).replace('_',' ').title()) for x in df.model]

def basic_plots(result_dir, out):
    path = result_dir / 'model_comparison.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Not found: {path}. Run train_handoff_csv.py first.')
    comp = pd.read_csv(path).sort_values('cv_macro_f1', ascending=False).reset_index(drop=True)
    labels = named(comp)
    for field, title, filename, percent in [
        ('cv_macro_f1','Trip-grouped CV candidate Macro F1','01_cv_macro_f1.png', False),
        ('cv_top1_accuracy','Trip-grouped CV observation Top-1 accuracy (proxy)','02_observation_top1_accuracy.png', True),
        ('cv_log_loss','Trip-grouped CV candidate log loss (lower is better)','03_cv_log_loss.png', False),
    ]:
        if field not in comp: continue
        y = comp[field].to_numpy(float)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        color = ['#2074ad' if i == 0 else '#a6b8c4' for i in range(len(y))]
        ax.barh(np.arange(len(y)), y * (100 if percent else 1), color=color)
        ax.set_yticks(range(len(y)), labels)
        ax.invert_yaxis()
        xmax = max(y)*(100 if percent else 1)*1.16
        ax.set_xlim(0, xmax if xmax else 1)
        ax.set_title(title, weight='bold', pad=14)
        ax.set_xlabel('Percent (%)' if percent else ('Log loss' if field == 'cv_log_loss' else 'Macro F1'))
        for i,v in enumerate(y):
            val = v*(100 if percent else 1)
            ax.text(val + xmax*0.012, i, f'{val:.2f}%' if percent else f'{val:.4f}', va='center', fontsize=10)
        ax.grid(axis='x', linestyle=':', alpha=.25)
        ax.set_axisbelow(True)
        fig.text(.01,.01,'PathSnap Team 14 | Trip-grouped cross-validation | Proxy candidate labels',fontsize=8,alpha=.7)
        save(fig,out,filename)
    sel = result_dir / 'selected_model.json'
    summ = result_dir / 'tuning_summary.json'
    trials = result_dir / 'optuna_trials.csv'
    if sel.exists() and summ.exists() and trials.exists():
        chosen = json.loads(sel.read_text())
        tuning = json.loads(summ.read_text())
        if tuning.get('experiment_id') != chosen.get('experiment_id'):
            print('[SKIP] Tuning plots: tuning_summary belongs to a different experiment. Run tune_handoff_csv.py for the current full-data model.')
        else:
            hist = pd.read_csv(trials)
            if 'state' in hist:
                hist = hist[hist['state'].astype(str).str.contains('COMPLETE',case=False,na=False)]
            if {'number','value'} <= set(hist.columns) and len(hist):
                hist = hist.sort_values('number')
                x = hist['number'].astype(int).to_numpy()+1
                val = hist['value'].astype(float).to_numpy()
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(x,val,'o',label='Trial Macro F1',markersize=5,color='#8ca5b5')
                ax.plot(x,np.maximum.accumulate(val),'-',label='Best-so-far Macro F1',lw=2.2,color='#176eac')
                ax.set_xlabel('Bayesian optimization trial number')
                ax.set_ylabel('Trip-grouped CV Macro F1')
                ax.set_title(f'Optuna (TPE) optimization history — {len(x)} completed trials',weight='bold')
                ax.set_xticks(x if len(x)<=25 else np.unique(np.linspace(1,len(x),10,dtype=int)))
                ax.grid(alpha=.2)
                ax.legend()
                save(fig,out,'04_bayesian_trial_history.png')
            base = tuning.get('base_cv_macro_f1')
            tuned = tuning.get('best_tuning_cv_macro_f1')
            if base is not None and tuned is not None:
                fig, ax = plt.subplots(figsize=(7,4.8))
                vals=[float(base), float(tuned)]
                ax.bar(['Untuned baseline','Best tuned trial'],vals,color=['#176eac','#8ca5b5'])
                ax.set_ylim(0,max(vals)*1.15)
                ax.set_ylabel('CV Macro F1')
                ax.set_title('Untuned vs Bayesian-tuned candidate model',weight='bold')
                for i,v in enumerate(vals): ax.text(i,v+max(vals)*.015,f'{v:.4f}',ha='center')
                ax.grid(axis='y',alpha=.2)
                ax.set_axisbelow(True)
                save(fig,out,'05_untuned_vs_tuned.png')
    else:
        print('[NOTE] Skipping Bayesian tuning figures until tuning_summary.json and optuna_trials.csv are present.')
    return comp

def grouped_cv_plots(out, result_dir, csv):
    # Import the exact factory and loader from your repository to avoid defining mismatched models.
    sys.path.insert(0,str(Path.cwd()/'src'))
    sys.path.insert(0,str(Path.cwd()/'scripts'))
    import joblib
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
    from pathsnap.modeling import model_factory
    from train_handoff_csv import load_table, with_fold_safe_imputation
    meta_path = Path('artifacts/handoff/best_model.joblib')
    if not meta_path.exists():
        raise FileNotFoundError(f'{meta_path} missing; run the main trainer first.')
    meta = joblib.load(meta_path)
    nfold = int(meta['group_cv_folds'])
    sample = int(meta['max_obs_per_trip'])
    if Path(meta['csv_source']).resolve() != csv.resolve():
        raise ValueError('CSV differs from the CSV used for the selected model. Pass --csv with its exact path.')
    df, feature_names, stats = load_table(csv, sample)
    if feature_names != meta['feature_names']:
        raise ValueError('Feature schema mismatch')
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df['label'].to_numpy(dtype=np.int8)
    groups = df['trip_id'].astype(str).to_numpy()
    obs = df['observation_index'].to_numpy()
    kf=GroupKFold(n_splits=nfold)
    folds=list(kf.split(X,y,groups))
    roc_curves=[]; pr_curves=[]; all_stats=[]
    for model_name,base in model_factory().items():
        print(f'[EVALUATING] {model_name}: {nfold} grouped folds on {len(df):,} rows',flush=True)
        p=np.empty(len(y),dtype=np.float32)
        for i,(tr,va) in enumerate(folds,1):
            m=with_fold_safe_imputation(clone(base))
            m.fit(X[tr],y[tr])
            p[va]=m.predict_proba(X[va])[:,1]
            print(f'    fold {i}/{nfold} complete',flush=True)
        pred=(p>=0.5).astype(np.int8)
        cm=confusion_matrix(y,pred,labels=[0,1])
        normalized=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(6.2,5.4))
        heat=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
        for i in range(2):
            for j in range(2):
                c=cm[i,j]
                ax.text(j,i,f'{c:,}\n({normalized[i,j]:.1%})',ha='center',va='center',fontsize=12,weight='bold',color='white' if normalized[i,j]>.55 else '#202a35')
        ax.set_xticks([0,1],['Pred: unmatched (0)','Pred: matched (1)'])
        ax.set_yticks([0,1],['True: unmatched (0)','True: matched (1)'])
        ax.set_xlabel('Predicted candidate label')
        ax.set_ylabel('Proxy reference candidate label')
        ax.set_title(DISPLAY.get(model_name,model_name)+' — out-of-fold confusion matrix',weight='bold',pad=13)
        fig.colorbar(heat,ax=ax,fraction=.046,pad=.04,label='Fraction of true class')
        save(fig,out,f'confusion_{safe_name(model_name)}.png')
        fpr,tpr,_=roc_curve(y,p)
        precision,recall,_=precision_recall_curve(y,p)
        roc_curves.append((DISPLAY.get(model_name,model_name),fpr,tpr,auc(fpr,tpr)))
        pr_curves.append((DISPLAY.get(model_name,model_name),recall,precision,average_precision_score(y,p)))
        # One reference-aligned candidate per GPS observation is the top-1 target.
        d=pd.DataFrame({'trip':groups,'obs':obs,'score':p,'label':y})
        pick=d.groupby(['trip','obs'],sort=False)['score'].idxmax()
        top1=float(d.loc[pick,'label'].mean())
        all_stats.append({'model':model_name,'oof_macro_f1':f1_score(y,pred,average='macro',zero_division=0),
                          'oof_top1_proxy':top1,'tn':int(cm[0,0]),'fp':int(cm[0,1]),
                          'fn':int(cm[1,0]),'tp':int(cm[1,1]),'roc_auc':auc(fpr,tpr),
                          'average_precision':average_precision_score(y,p)})
    summary=pd.DataFrame(all_stats).sort_values('oof_macro_f1',ascending=False)
    summary.to_csv(out/'oof_model_metrics.csv',index=False)
    print('[SAVED]',out/'oof_model_metrics.csv')
    for curves,title,filename,xlabel,ylabel in [
        (roc_curves,'Out-of-fold ROC curves — all five models','06_roc_comparison.png','False Positive Rate','True Positive Rate'),
        (pr_curves,'Out-of-fold Precision–Recall curves — all five models','07_pr_comparison.png','Recall','Precision'),
    ]:
        fig,ax=plt.subplots(figsize=(8,6.5))
        for name,x,ys,value in curves:
            ax.plot(x,ys,label=f'{name} (area={value:.3f})',linewidth=1.8)
        if filename.startswith('06'): ax.plot([0,1],[0,1],'--',color='#aaa',lw=1)
        else: ax.axhline(float(np.mean(y)),linestyle=':',color='#aaa',lw=1,label='Positive-class prevalence')
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title,xlim=(0,1),ylim=(0,1.02))
        ax.legend(loc='best',fontsize=9)
        ax.grid(alpha=.15)
        save(fig,out,filename)
    print('\nOUT-OF-FOLD METRICS:\n',summary.to_string(index=False))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results/handoff',help='Where model_comparison.csv and tuning files live')
    parser.add_argument('--output',default='results/presentation_graphs')
    parser.add_argument('--confusion',action='store_true',help='Train honest grouped out-of-fold predictions for all 5 confusion matrices, ROC and PR; takes more compute')
    parser.add_argument('--csv',default='data/handoff/candidate_features_handoff.csv')
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    basic_plots(Path(args.results),out)
    if args.confusion:
        grouped_cv_plots(out,Path(args.results),Path(args.csv).resolve())
    else:
        print('[NOTE] Add --confusion to compute 5 real grouped-CV confusion matrices + ROC/PR curves.')

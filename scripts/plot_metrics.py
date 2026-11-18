"""Generate actual-result comparison plots from results/model_comparison.csv."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

path = Path('results/model_comparison.csv')
if not path.exists():
    raise SystemExit('Run scripts/train_models.py before generating figures')
df = pd.read_csv(path)
for column, title, name in [
    ('cv_macro_f1', 'Grouped CV: Candidate Macro F1', 'cv_macro_f1.png'),
    ('student_test_observation_top1_accuracy_proxy', 'Student Test: Top-1 Proxy Accuracy', 'test_top1.png'),
    ('student_test_candidate_macro_f1', 'Student Test: Candidate Macro F1', 'test_macro_f1.png'),
    ('student_test_candidate_log_loss', 'Student Test: Candidate Log Loss', 'test_log_loss.png')]:
    if column not in df:
        continue
    ax = df.plot(x='model', y=column, kind='bar', legend=False, title=title, rot=25)
    ax.set_ylabel(column)
    ax.set_xlabel('Model')
    ax.figure.tight_layout()
    ax.figure.savefig(path.parent / name, dpi=150)
    plt.close(ax.figure)

"""Train-only grouped cross-validation of feature classifiers; no test-driven tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from foodquality8.evaluation_cache import align_features
from foodquality8.robust_views import LEGACY
from foodquality8.food9 import FOODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=a.base/'split.json';rows=json.loads(manifest.read_text(encoding='utf8'))
    features=align_features(a.base,LEGACY,rows)
    y=np.array([FOODS.index(r['food'])*2+int(r['condition']=='spoiled') for r in rows])
    groups=np.array([r.get('duplicate_group',r['sha256']) for r in rows]);train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    candidates={'ridge_1':RidgeClassifier(alpha=1,class_weight='balanced'),
                'ridge_10':RidgeClassifier(alpha=10,class_weight='balanced'),
                'logistic_01':LogisticRegression(C=.1,max_iter=300,class_weight='balanced',solver='lbfgs')}
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(features[train],y[train],groups[train]))
    scores={};started=time.perf_counter()
    for name,estimator in candidates.items():
        scores[name]=[]
        for fit,valid in folds:
            assert not set(groups[train[fit]]) & set(groups[train[valid]])
            pipeline=make_pipeline(StandardScaler(),clone(estimator));pipeline.fit(features[train[fit]],y[train[fit]])
            score=float(f1_score(y[train[valid]],pipeline.predict(features[train[valid]]),average='macro',labels=range(18),zero_division=0))
            scores[name].append(score);print(name,score,flush=True)
    selected=max(scores,key=lambda key:np.mean(scores[key]));pipeline=make_pipeline(StandardScaler(),clone(candidates[selected]));pipeline.fit(features[train],y[train]);joblib.dump(pipeline,a.output/'classifier.joblib')
    report=dict(selected=selected,cv=scores,selection_scope='3 grouped folds of training data only',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),seconds=time.perf_counter()-started,evaluation={})
    names=[f'{f}_{s}' for f in FOODS for s in ('fresh','spoiled')]
    for split in ('val','test'):
        ids=[i for i,r in enumerate(rows) if r['split']==split];prediction=pipeline.predict(features[ids]);truth=y[ids]
        report['evaluation'][split]=dict(samples=len(ids),food_accuracy=float(np.mean(prediction//2==truth//2)),joint_accuracy=float(np.mean(prediction==truth)),macro_f1=float(f1_score(truth,prediction,labels=range(18),average='macro',zero_division=0)),per_class=classification_report(truth,prediction,labels=range(18),target_names=names,output_dict=True,zero_division=0),confusion_matrix=confusion_matrix(truth,prediction,labels=range(18)).tolist())
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(selected,report['evaluation']['val']['macro_f1'],flush=True)


if __name__=='__main__':main()

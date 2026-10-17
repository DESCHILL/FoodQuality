"""Compare regularized linear heads and cosine neighbours on fixed source splits."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score,f1_score
from foodquality8.food9 import FOODS,encoder
from foodquality8.model_data import write_json

def main():
    rows=json.loads((ROOT/'workspace/sic_merged/split.json').read_text(encoding='utf8'))
    features=np.load(ROOT/'workspace/sic_merged/features.npz')['features'][:,:1280]
    x=normalize(features)
    y=np.array([FOODS.index(r['food'])*2+int(r['condition']=='spoiled') for r in rows])
    train=np.array([r['split']=='train' for r in rows]);val=np.array([r['split']=='val' for r in rows])
    models=[]
    for c in (.1,1,10,100):
        model=LogisticRegression(C=c,max_iter=600,class_weight='balanced').fit(x[train],y[train])
        models.append((f'cosine_logistic_C{c}',model))
    for k in (3,9,21):
        models.append((f'cosine_knn_k{k}',KNeighborsClassifier(n_neighbors=k,weights='distance',metric='cosine').fit(x[train],y[train])))
    report=[]
    for name,model in models:
        pred=model.predict(x[val]);score=f1_score(y[val],pred,average='macro')
        report.append(dict(algorithm=name,val_accuracy=accuracy_score(y[val],pred),val_macro_f1=score))
    selected=max(range(len(report)),key=lambda i:report[i]['val_macro_f1'])
    name,model=models[selected]
    # Test is reported once after selecting with validation.
    test=np.array([r['split']=='test' for r in rows]);pred=model.predict(x[test])
    output=dict(validation=report,selected=name,test_accuracy=accuracy_score(y[test],pred),test_macro_f1=f1_score(y[test],pred,average='macro'))
    write_json(ROOT/'workspace/algorithm-benchmark.json',output)
    if isinstance(model,LogisticRegression):
        np.savez_compressed(ROOT/'models/sic/cosine-head.npz',weight=model.coef_,bias=model.intercept_,classes=model.classes_)
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()

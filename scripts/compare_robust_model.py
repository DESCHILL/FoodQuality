"""Compare matching source holdouts, using each checkpoint's own preprocessing."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,recall_score
from foodquality8.food9 import FOODS
from foodquality8.robust_views import LEGACY,feature_signature
from foodquality8.model_data import write_json


def predict_heads(checkpoint,features,legacy_features=None):
    x=((torch.as_tensor(features)-checkpoint['mean'])/checkpoint['std']).clamp(-8,8)
    with torch.inference_mode():
        food=(x@checkpoint['food_head']['weight'].T+checkpoint['food_head']['bias']).argmax(1)
        from foodquality8.quality_ensemble import condition_logits
        logits=condition_logits(checkpoint,torch.as_tensor(features),None if legacy_features is None else torch.as_tensor(legacy_features))
        quality=logits[torch.arange(len(x)),food].argmax(1)
    return (food*2+quality).numpy()


def metrics(truth,prediction):
    labels=sorted(set(truth))
    return dict(samples=len(truth),food_accuracy=float(accuracy_score(truth//2,prediction//2)),
                joint_accuracy=float(accuracy_score(truth,prediction)),
                macro_f1=float(f1_score(truth,prediction,labels=labels,average='macro',zero_division=0)),
                spoiled_recall=float(recall_score(truth,prediction,labels=[x for x in labels if x%2],average='macro',zero_division=0)))


def passes(baseline,candidate):
    return (candidate['macro_f1']>baseline['macro_f1']
            and candidate['food_accuracy']>=baseline['food_accuracy']
            and candidate['spoiled_recall']>=baseline['spoiled_recall'])


def load_features(base,checkpoint):
    rows=json.loads((base/'split.json').read_text(encoding='utf8'))
    cache=np.load(base/'features.npz')
    recipe=checkpoint.get('preprocessing',LEGACY)
    allowed={feature_signature(rows,recipe)}
    if recipe==LEGACY:
        allowed.add(hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in rows]).encode()).hexdigest())
    if str(cache['signature']) not in allowed:
        raise ValueError('Feature cache recipe or dataset mismatch: '+str(base))
    return rows,cache['features']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,default=ROOT/'models/sic/robust9-candidate.json')
    p.add_argument('--base',type=Path,default=ROOT/'workspace/robust9_v1')
    p.add_argument('--original-base',type=Path,default=ROOT/'workspace/tomato_v3')
    p.add_argument('--output',type=Path,default=ROOT/'workspace/robust9_v1/comparison.json')
    p.add_argument('--reference-manifest',type=Path,default=ROOT/'workspace/tomato_v3/split.json')
    args=p.parse_args();torch.set_num_threads(4)
    live=json.loads((ROOT/'models/consensus.json').read_text(encoding='utf8'))
    entries={'baseline':Path(live['quality_registry']),'candidate':args.candidate}
    predictions={};reference=None
    for name,path in entries.items():
        registry=json.loads(path.read_text(encoding='utf8'))
        if hashlib.sha256(Path(registry['weights']).read_bytes()).hexdigest()!=registry['sha256']:
            raise ValueError('Checkpoint checksum mismatch.')
        checkpoint=torch.load(registry['weights'],weights_only=True,map_location='cpu')
        feature_base=(args.original_base if name=='baseline' and checkpoint.get('preprocessing',LEGACY)==LEGACY else args.base)
        rows,x=load_features(feature_base,checkpoint)
        ids=[i for i,r in enumerate(rows) if r['split'] in ('val','test') and not r.get('derived_from')]
        selected=[rows[i] for i in ids]
        if reference is not None and [(r['sha256'],r['split']) for r in reference]!=[(r['sha256'],r['split']) for r in selected]:
            raise ValueError('Baseline and candidate holdouts differ.')
        legacy_features=None
        if checkpoint.get('legacy_quality'):
            legacy_rows,legacy_x=load_features(args.original_base,{'preprocessing':LEGACY})
            lookup={r['sha256']:i for i,r in enumerate(legacy_rows)}
            legacy_features=legacy_x[[lookup[r['sha256']] for r in selected]]
        reference=selected;predictions[name]=predict_heads(checkpoint,x[ids],legacy_features)
    truth=np.array([FOODS.index(r['food'])*2+int(r['condition']=='spoiled') for r in reference])
    masks={s:np.array([r['split']==s for r in reference]) for s in ('val','test')}
    masks['val_original']=np.array([r['split']=='val' and r.get('source')!='AgriFreshNET v1' for r in reference])
    if args.reference_manifest.exists():
        locked=json.loads(args.reference_manifest.read_text(encoding='utf8'))
        locked_val={r['sha256'] for r in locked if r['split']=='val'}
        current_val={r['sha256'] for r in reference if r['split']=='val'}
        if not locked_val.issubset(current_val):raise ValueError('Locked validation images are missing or moved.')
        masks['val_locked']=np.array([r['split']=='val' and r['sha256'] in locked_val for r in reference])
        if current_val-locked_val:
            masks['val_added']=np.array([r['split']=='val' and r['sha256'] not in locked_val for r in reference])
    for food in FOODS:
        masks['val_'+food]=np.array([r['split']=='val' and r['food']==food for r in reference])
    results={name:{key:metrics(truth[mask],pred[mask]) for key,mask in masks.items()} for name,pred in predictions.items()}
    passed=passes(results['baseline']['val'],results['candidate']['val'])
    passed=passed and results['candidate']['val_original']['macro_f1']>=results['baseline']['val_original']['macro_f1']-.005
    if 'val_locked' in masks:
        before=results['baseline']['val_locked'];after=results['candidate']['val_locked']
        passed=passed and after['macro_f1']>=before['macro_f1']-.005 and after['food_accuracy']>=before['food_accuracy']
    output=dict(passed=passed,selection_split='val',metrics=results,
                candidate_registry_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                baseline_registry_sha256=live['quality_registry_sha256'],
                rule='Validation macro F1 improves, food accuracy and spoiled recall do not decline; original-source and locked validation F1 regression <=0.5 percentage point; locked food accuracy does not decline.',
                limitation='Internal source-labelled data; transformed images are not new independent examples.')
    write_json(args.output,output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()

"""Select one global quality mixture on validation, retaining the detail-view food head."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from foodquality8.food9 import FOODS
from foodquality8.robust_views import LEGACY
from foodquality8.model_data import write_json
from scripts.compare_robust_model import load_features,predict_heads,metrics,passes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--base',type=Path,default=ROOT/'workspace/robust9_v1')
    p.add_argument('--original-base',type=Path,default=ROOT/'workspace/tomato_v3')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    if args.checkpoint.exists():raise ValueError('Choose a new ensemble checkpoint path.')
    live=json.loads((ROOT/'models/consensus.json').read_text(encoding='utf8'))
    oldreg=json.loads(Path(live['quality_registry']).read_text(encoding='utf8'))
    registry=json.loads(args.candidate.read_text(encoding='utf8'))
    for r in (registry,oldreg):
        if hashlib.sha256(Path(r['weights']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Changed checkpoint.')
    old=torch.load(oldreg['weights'],weights_only=True,map_location='cpu')
    new=torch.load(registry['weights'],weights_only=True,map_location='cpu')
    if not all(torch.equal(value,old['encoder'][key]) for key,value in new['encoder'].items()):
        raise ValueError('Mixture requires identical frozen encoders.')
    reference=old.get('legacy_quality')
    if reference is None:
        if old.get('preprocessing',LEGACY)!=LEGACY:raise ValueError('No compatible legacy quality head.')
        reference={k:old[k] for k in ('quality_head','mean','std')}
    legacy_rows,legacy_x=load_features(args.original_base,{'preprocessing':LEGACY})
    rows,x=load_features(args.base,new);lookup={r['sha256']:i for i,r in enumerate(rows)}
    selected=[r for r in legacy_rows if r['split'] in ('val','test')]
    original_lookup={r['sha256']:i for i,r in enumerate(legacy_rows)}
    raw=x[[lookup[r['sha256']] for r in selected]]
    legacy_raw=legacy_x[[original_lookup[r['sha256']] for r in selected]]
    truth=np.array([FOODS.index(r['food'])*2+int(r['condition']=='spoiled') for r in selected])
    val=np.array([r['split']=='val' for r in selected]);test=~val
    baseline_raw=legacy_raw if old.get('preprocessing',LEGACY)==LEGACY else raw
    baseline=metrics(truth[val],predict_heads(old,baseline_raw[val],legacy_raw[val]))
    trials=[];best=None
    for weight in np.linspace(0,1,11):
        new['legacy_quality']=dict(reference,weight=float(weight),source_sha256=oldreg['sha256'])
        result=metrics(truth[val],predict_heads(new,raw[val],legacy_raw[val]))
        trials.append(dict(weight=float(weight),**result))
        if passes(baseline,result) and (best is None or result['macro_f1']>best['macro_f1']):
            best=trials[-1]
    report=dict(baseline=baseline,trials=trials,selected=best,selection_split='val')
    write_json(args.checkpoint.with_suffix('.selection.json'),report)
    if best is None:
        print('No quality mixture passed validation; active model unchanged.');return
    new['legacy_quality']=dict(reference,weight=best['weight'],source_sha256=oldreg['sha256'])
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save(new,args.checkpoint)
    result=metrics(truth[test],predict_heads(new,raw[test],legacy_raw[test]))
    registry.update(weights=str(args.checkpoint.resolve()),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    quality_ensemble_weight=best['weight'],test=result)
    write_json(args.output,registry)
    print(json.dumps(dict(selected=best,test=result),indent=2))


if __name__=='__main__':main()

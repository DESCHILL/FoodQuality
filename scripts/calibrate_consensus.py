"""Fit temperatures on validation and publish a versioned three-source candidate."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from foodquality8.food9 import FOODS
from foodquality8.model_data import write_json

def temperature(logits,target):
    def loss(logt):return float(torch.nn.functional.cross_entropy(logits/np.exp(logt),target))
    fit=minimize_scalar(loss,bounds=(np.log(.25),np.log(5)),method='bounded')
    return float(np.exp(fit.x))

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=ROOT/'workspace/sic_merged')
    parser.add_argument('--legacy-base',type=Path,default=ROOT/'workspace/sic_merged')
    parser.add_argument('--quality-registry',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'models/consensus-candidate.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    registry=json.loads((ROOT/'models/fusion.json').read_text(encoding='utf8'))
    qp=(args.quality_registry or Path(registry['quality_registry'])).resolve();qreg=json.loads(qp.read_text(encoding='utf8'))
    ck=torch.load(qreg['weights'],weights_only=True,map_location='cpu')
    rows=json.loads((args.base/'split.json').read_text(encoding='utf8'))
    ids=[i for i,r in enumerate(rows) if r['split']=='val']
    from foodquality8.robust_views import LEGACY,feature_signature
    cache=np.load(args.base/'features.npz')
    expected=feature_signature(rows,ck.get('preprocessing',LEGACY))
    legacy_signature=hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in rows]).encode()).hexdigest()
    if str(cache['signature'])!=expected and not (ck.get('preprocessing',LEGACY)==LEGACY and str(cache['signature'])==legacy_signature):
        raise ValueError('Feature cache does not match checkpoint preprocessing or dataset.')
    x=torch.from_numpy(cache['features'][ids])
    y=torch.tensor([FOODS.index(rows[i]['food']) for i in ids]);q=torch.tensor([int(rows[i]['condition']=='spoiled') for i in ids])
    normalized=((x-ck['mean'])/ck['std']).clamp(-8,8)
    fl=normalized@ck['food_head']['weight'].T+ck['food_head']['bias']
    ql=(normalized@ck['quality_head']['weight'].T+ck['quality_head']['bias']).reshape(-1,9,2)[torch.arange(len(ids)),y]
    sicpath=ROOT/'models/sic/condition.pth';sic=torch.load(sicpath,weights_only=True,map_location='cpu')
    legacy_rows=json.loads((args.legacy_base/'split.json').read_text(encoding='utf8'))
    legacy_indices={r['sha256']:i for i,r in enumerate(legacy_rows) if r['split']=='val'}
    legacy_cache=np.load(args.legacy_base/'features.npz')
    legacy_expected=hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in legacy_rows]).encode()).hexdigest()
    if str(legacy_cache['signature']) not in (legacy_expected,feature_signature(legacy_rows,LEGACY)):
        raise ValueError('SIC requires original 224-pixel validation features.')
    sx=torch.from_numpy(legacy_cache['features'][[legacy_indices[rows[i]['sha256']] for i in ids],:1280])
    sl=sx@sic['model_state_dict']['classifier.1.weight'].T+sic['model_state_dict']['classifier.1.bias']
    names=sic['class_names'];foods=sorted({n.split('__')[0] for n in names})
    sfl=torch.stack([sl[:,[i for i,n in enumerate(names) if n.split('__')[0]==f]].logsumexp(1) for f in foods],1)
    valid=[j for j,i in enumerate(ids) if rows[i]['food'] in foods]
    sy=torch.tensor([foods.index(rows[ids[j]]['food']) for j in valid])
    sql=[];sq=[]
    for j in valid:
        food=rows[ids[j]]['food'];indices=[names.index(food+'__'+s) for s in ('normal_appearance','visible_defect')]
        sql.append(sl[j,indices]);sq.append(int(rows[ids[j]]['condition']=='spoiled'))
    cached=json.loads((ROOT/'workspace/sic_merged/fusion-val.json').read_text(encoding='utf8'))['records']
    scores=[];correct=[]
    for record in cached:
        for pred in record['predicted']:
            if pred.get('bbox') is None:continue
            raw=np.clip(pred['detection_score'],1e-6,1-1e-6)
            scores.append([np.log(raw/(1-raw))]);correct.append(int(pred['food'].lower().replace(' ','_')==record['food']))
    platt=LogisticRegression(C=1).fit(scores,correct)
    calibration=dict(foodquality_temperature=temperature(fl,y),foodquality_quality_temperature=temperature(ql,q),
        sic_temperature=temperature(sfl[valid],sy),sic_quality_temperature=temperature(torch.stack(sql),torch.tensor(sq)),
        yolo_platt=[float(platt.coef_[0,0]),float(platt.intercept_[0])],
        scope='Source-labelled validation; YOLO calibrated for image-level food agreement on retained boxes >=0.4, not localization IoU; below 0.4 score is extrapolated.')
    calibration['sic_quality_by_food']={}
    for food in foods:
        positions=[k for k,j in enumerate(valid) if rows[ids[j]]['food']==food]
        if len(positions)>=50:
            calibration['sic_quality_by_food'][food]=temperature(torch.stack(sql)[positions],torch.tensor(sq)[positions])
    config=dict(mode='three_source_consensus',quality_registry=str(qp),sic_classifier=str(sicpath),detector=registry['detector'],
        calibration=calibration,foods=FOODS,quality_threshold=.8)
    for key in ('quality_registry','sic_classifier','detector'):
        config[key+'_sha256']=hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    config['preprocessing']=ck.get('preprocessing',LEGACY)
    config['validation_manifest_sha256']=hashlib.sha256((args.base/'split.json').read_bytes()).hexdigest()
    write_json(args.output,config)
    print(json.dumps(calibration,indent=2),flush=True)

if __name__=='__main__':main()

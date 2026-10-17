"""Compare classifiers on identical images; activation uses validation only."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,recall_score
from foodquality8.food9 import FOODS
from foodquality8.model_data import write_json

def main():
    torch.set_num_threads(4)
    base=ROOT/'workspace/sic_merged'
    rows=json.loads((base/'split.json').read_text(encoding='utf8'))
    features=torch.from_numpy(np.load(base/'features.npz')['features'])
    truth=np.array([FOODS.index(r['food'])*2+int(r['condition']=='spoiled') for r in rows])
    predictions={};registries={}
    entries=[('foodquality_original',ROOT/'models/food9.json'),('merged_candidate',ROOT/'models/sic/candidate.json')]
    deployed=ROOT/'models/fusion.json'
    if deployed.exists():
        entries.append(('deployed',Path(json.loads(deployed.read_text(encoding='utf8'))['quality_registry'])))
    for name,path in entries:
        registry=json.loads(path.read_text(encoding='utf8'));registries[name]=registry
        ck=torch.load(registry['weights'],weights_only=True,map_location='cpu')
        x=((features-ck['mean'])/ck['std']).clamp(-8,8)
        f=x@ck['food_head']['weight'].T+ck['food_head']['bias']
        q=(x@ck['quality_head']['weight'].T+ck['quality_head']['bias']).reshape(-1,9,2)
        food=f.argmax(1);condition=q[torch.arange(len(x)),food].argmax(1)
        predictions[name]=(food*2+condition).numpy()
    ck=torch.load(ROOT/'models/sic/condition.pth',weights_only=True,map_location='cpu')
    reference=torch.load(registries['foodquality_original']['weights'],weights_only=True,map_location='cpu')
    if not all(torch.equal(v,reference['encoder'][k]) for k,v in ck['model_state_dict'].items() if not k.startswith('classifier.')):
        raise ValueError('SIC backbone differs: shared cached features cannot be used for comparison.')
    logits=features[:,:1280]@ck['model_state_dict']['classifier.1.weight'].T+ck['model_state_dict']['classifier.1.bias']
    names=ck['class_names'];mapped=[]
    for idx in logits.argmax(1).tolist():
        food,state=names[idx].split('__')
        mapped.append(FOODS.index(food)*2+int(state=='visible_defect') if food in FOODS else -2)
    predictions['sic_original']=np.array(mapped)
    def evaluate(pred,mask):
        y=truth[mask];p=pred[mask]
        return dict(samples=len(y),food_accuracy=accuracy_score(y//2,p//2),joint_accuracy=accuracy_score(y,p),
            macro_f1=f1_score(y,p,labels=sorted(set(y)),average='macro',zero_division=0),
            spoiled_recall=recall_score(y,p,labels=sorted(v for v in set(y) if v%2),average='macro',zero_division=0))
    masks={}
    for split in ('val','test'):
        masks[split]=np.array([r['split']==split for r in rows])
        masks[split+'_common8']=np.array([r['split']==split and r['food']!='mutton' for r in rows])
        for source in ('FoodClass V2','AgML raw / bdd69gyhv8.1'):
            masks[split+'_'+source.split()[0]]=np.array([r['split']==split and r['source']==source for r in rows])
    comparison={name:{key:evaluate(p,mask) for key,mask in masks.items()} for name,p in predictions.items()}
    baseline_name='deployed' if 'deployed' in comparison else 'foodquality_original'
    a=comparison['merged_candidate'];b=comparison[baseline_name]
    passed=(a['val']['macro_f1']>b['val']['macro_f1'] and
        a['val_FoodClass']['macro_f1']>=b['val_FoodClass']['macro_f1']-.005 and
        a['val']['spoiled_recall']>=b['val']['spoiled_recall'])
    selected='merged_candidate' if passed else baseline_name
    write_json(ROOT/'models/sic/quality-selected.json',registries[selected])
    write_json(base/'comparison.json',dict(selected=selected,metrics=comparison,
        selection='Validation macro-F1 improves; old-source F1 regression <=0.5 percentage point; spoiled recall cannot decline.',
        limitation='Internal source-labelled diagnostic, not independent real-world evaluation; overlap with SIC training unknown. Visible_defect is compared as abnormal against source Rotten labels, not a universal spoiled equivalence.'))
    print(json.dumps(dict(selected=selected,metrics=comparison),indent=2),flush=True)
    detector=ROOT/'models/sic/detector.pt';quality=ROOT/'models/sic/quality-selected.json'
    write_json(ROOT/'models/sic/fusion-candidate.json',dict(detector=str(detector),detector_sha256=hashlib.sha256(detector.read_bytes()).hexdigest(),
        quality_registry=str(quality),quality_registry_sha256=hashlib.sha256(quality.read_bytes()).hexdigest(),
        foods=FOODS,mode='sic_foodquality_fusion',quality_source=registries[selected]['source']))

if __name__=='__main__':main()

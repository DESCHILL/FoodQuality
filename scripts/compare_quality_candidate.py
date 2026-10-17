"""Evaluate a source-expanded candidate against the deployed classifier on identical splits."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,recall_score
from foodquality8.food9 import FOODS
from foodquality8.model_data import write_json

def main():
    torch.set_num_threads(4)
    base=ROOT/'workspace/tomato_v3';rows=json.loads((base/'split.json').read_text(encoding='utf8'))
    x=torch.from_numpy(np.load(base/'features.npz')['features'])
    y=np.array([FOODS.index(r['food'])*2+int(r['condition']=='spoiled') for r in rows])
    deployed=json.loads((ROOT/'models/fusion.json').read_text(encoding='utf8'))
    output={}
    for name,path in [('deployed',Path(deployed['quality_registry'])),('candidate',ROOT/'models/sic/tomato-candidate.json')]:
        reg=json.loads(path.read_text(encoding='utf8'));ck=torch.load(reg['weights'],weights_only=True,map_location='cpu')
        normalized=((x-ck['mean'])/ck['std']).clamp(-8,8)
        fp=(normalized@ck['food_head']['weight'].T+ck['food_head']['bias']).argmax(1)
        qp=(normalized@ck['quality_head']['weight'].T+ck['quality_head']['bias']).reshape(-1,9,2)[torch.arange(len(x)),fp].argmax(1)
        pred=(fp*2+qp).numpy();metrics={}
        for split in ('val','test'):
            for group in ('all','original','tomato'):
                ids=np.array([r['split']==split and (group=='all' or (group=='original' and r.get('source')!='AgriFreshNET v1') or (group=='tomato' and r['food']=='tomato')) for r in rows])
                labels=sorted(set(y[ids]))
                metrics[split+'_'+group]=dict(samples=int(ids.sum()),accuracy=accuracy_score(y[ids],pred[ids]),
                    macro_f1=f1_score(y[ids],pred[ids],labels=labels,average='macro',zero_division=0),
                    spoiled_recall=recall_score(y[ids],pred[ids],labels=[v for v in labels if v%2],average='macro',zero_division=0))
        output[name]=metrics
    a=output['candidate'];b=output['deployed']
    passed=a['val_all']['macro_f1']>b['val_all']['macro_f1'] and a['val_original']['macro_f1']>=b['val_original']['macro_f1']-.005 and a['val_all']['spoiled_recall']>=b['val_all']['spoiled_recall']
    output['selected']='candidate' if passed else 'deployed'
    write_json(base/'comparison.json',output)
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()

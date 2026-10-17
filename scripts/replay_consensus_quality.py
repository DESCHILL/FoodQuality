"""Re-evaluate only the deterministic quality policy on recorded model outputs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from foodquality8.consensus import decide_quality
from foodquality8.food9 import FOODS
from foodquality8.model_data import write_json

def main():
    path=ROOT/'workspace/consensus-evaluation.json'
    report=json.loads(path.read_text(encoding='utf8'))
    config=json.loads((ROOT/'models/consensus.json').read_text(encoding='utf8'))
    rows=report['records']
    for row in rows:
        p=row['prediction'];condition,reason=decide_quality(p['food'],p['quality_evidence'],config['quality_threshold'])
        if condition=='defect':condition='unknown';reason='Bất thường ngoại quan chưa chứng minh hỏng.'
        if p['food'] not in FOODS or p['detection_score']<.6:condition='unknown'
        p.update(condition=condition,reason=reason)
    accepted=[r for r in rows if r['prediction']['condition']!='unknown']
    correct=sum(r['prediction']['food']==r['expected_food'] and r['prediction']['condition']==r['expected_condition'] for r in accepted)
    report['summary'].update(accepted=len(accepted),correct_accepted=correct,
        accepted_accuracy=correct/max(1,len(accepted)),coverage=len(accepted)/len(rows),
        policy_replayed=True,note='Same stored outputs from all three models; only deterministic defect/spoiled compatibility changed. Internal development benchmark, not an untouched external test.')
    write_json(ROOT/'workspace/consensus-evaluation-final.json',report)
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()

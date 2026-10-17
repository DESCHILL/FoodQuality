"""Stratified end-to-end smoke benchmark; test images never fit calibration."""
import json,random,sys,time
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from foodquality8.consensus import ConsensusPredictor
from foodquality8.model_data import write_json

def main():
    model=ConsensusPredictor(ROOT/'models/consensus-candidate.json')
    rows=json.loads((ROOT/'workspace/sic_merged/split.json').read_text(encoding='utf8'))
    groups=defaultdict(list)
    for row in rows:
        if row['split']=='test':groups[(row['food'],row['condition'])].append(row)
    chosen=[]
    for items in groups.values():
        random.Random(194).shuffle(items);chosen+=items if '--full' in sys.argv else items[:5]
    records=[];started=time.perf_counter()
    for i,row in enumerate(chosen):
        with Image.open(row['path']) as im:pred=model.predict(im)[0]
        assert len(pred['source_scores'])==3
        records.append(dict(expected_food=row['food'],expected_condition=row['condition'],prediction=pred))
        if (i+1)%20==0:print('Consensus test',i+1,'/',len(chosen),flush=True)
    accepted=[r for r in records if r['prediction']['condition']!='unknown']
    correct=sum(r['prediction']['food']==r['expected_food'] and r['prediction']['condition']==r['expected_condition'] for r in accepted)
    summary=dict(samples=len(records),food_accuracy=sum(r['prediction']['food']==r['expected_food'] for r in records)/len(records),
        accepted=len(accepted),correct_accepted=correct,accepted_accuracy=correct/max(1,len(accepted)),coverage=len(accepted)/len(records),
        seconds=time.perf_counter()-started,scope='Full internal test split; no real-world accuracy claim.' if '--full' in sys.argv else 'Stratified smoke subset, not the full held-out benchmark or real-world accuracy.')
    write_json(ROOT/'workspace/consensus-evaluation.json',dict(summary=summary,records=records))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

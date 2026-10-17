"""Evaluate real detector/crop path, counting abstentions and spurious detections."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from foodquality8.fusion import FusionPredictor
from foodquality8.model_data import write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--registry',type=Path,default=ROOT/'models/sic/fusion-candidate.json')
    args=parser.parse_args()
    source=ROOT/'workspace/sic_merged'
    rows=[r for r in json.loads((source/'split.json').read_text(encoding='utf8')) if r['split']==args.split]
    model=FusionPredictor(args.registry)
    records=[]
    for i,row in enumerate(rows):
        with Image.open(row['path']) as im:
            im=im.convert('RGB')
            baseline=model.quality.predict(im)
            predicted=model.predict(im)
        records.append(dict(sha256=row['sha256'],food=row['food'],condition=row['condition'],source=row['source'],
            baseline=baseline,predicted=predicted))
        if (i+1)%100==0:print('Fusion',args.split,i+1,'/',len(rows),flush=True)
    def metrics(key):
        accepted=correct=food_correct=claims=wrong_claims=0
        for r in records:
            preds=r[key]
            # Highest detection score is the single-item benchmark prediction.
            p=max(preds,key=lambda p:p['detection_score'])
            food_correct+=p['food']==r['food']
            if p['condition']!='unknown':
                accepted+=1;correct+=p['food']==r['food'] and p['condition']==r['condition']
            for p in preds:
                if p['condition']!='unknown':
                    claims+=1;wrong_claims+=not(p['food']==r['food'] and p['condition']==r['condition'])
        return dict(samples=len(records),food_accuracy=food_correct/len(records),accepted=accepted,
            coverage=accepted/len(records),accepted_accuracy=correct/max(1,accepted),
            correct_and_accepted_rate=correct/len(records),total_state_claims=claims,incorrect_state_claims=wrong_claims)
    report=dict(split=args.split,config_sha256=hashlib.sha256(args.registry.read_bytes()).hexdigest(),baseline=metrics('baseline'),fusion=metrics('predicted'),
        limitation='Single-food source labels; no bounding-box ground truth. Not a localization mAP evaluation or real-world independent test.',records=records)
    write_json(source/f'fusion-{args.split}.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__':main()

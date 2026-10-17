"""Choose routing on validation only and publish a pinned, rollbackable bundle."""
import copy,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from foodquality8.model_data import write_json

def routing_metrics(records,primary_first):
    accepted=correct=food_correct=0
    for record in records:
        predictions=record['baseline'] if primary_first and not record['baseline'][0].get('tentative') else record['predicted']
        pred=max(predictions,key=lambda p:p['detection_score'])
        food_correct+=pred['food']==record['food']
        if pred['condition']!='unknown':
            accepted+=1;correct+=pred['food']==record['food'] and pred['condition']==record['condition']
    return dict(samples=len(records),food_accuracy=food_correct/len(records),coverage=accepted/len(records),
        accepted_accuracy=correct/max(1,accepted),correct_and_accepted_rate=correct/len(records))

def main():
    path=ROOT/'workspace/sic_merged/fusion-val.json'
    validation=json.loads(path.read_text(encoding='utf8'))
    config=json.loads((ROOT/'models/sic/fusion-candidate.json').read_text(encoding='utf8'))
    if validation.get('config_sha256')!=hashlib.sha256((ROOT/'models/sic/fusion-candidate.json').read_bytes()).hexdigest():
        raise ValueError('Validation does not match the candidate configuration.')
    first=routing_metrics(validation['records'],True)
    detector_first=routing_metrics(validation['records'],False)
    # Prefer the route that provides more correct accepted answers, with a precision guard.
    passed=(first['correct_and_accepted_rate']>=detector_first['correct_and_accepted_rate'] and
        first['accepted_accuracy']>=validation['baseline']['accepted_accuracy']-.005)
    config['routing']='classification_first' if passed else 'detector_first'
    if not passed:
        raise ValueError('Combined routing has not met validation gate; live website unchanged.')
    quality=Path(config['quality_registry']);digest=hashlib.sha256(quality.read_bytes()).hexdigest()
    pinned=ROOT/'models/sic'/f'quality-{digest[:16]}.json'
    pinned.write_bytes(quality.read_bytes());config['quality_registry']=str(pinned)
    config['quality_registry_sha256']=digest
    config['routing_validation']=first
    active=ROOT/'models/fusion.json'
    if active.exists():
        (ROOT/'models/sic'/f'fusion-backup-{time.time_ns()}.json').write_bytes(active.read_bytes())
    write_json(ROOT/'workspace/sic_merged/routing-selection.json',dict(selected=config['routing'],
        classification_first=first,detector_first=detector_first,selection_split='val'))
    temporary=active.with_suffix('.pending.json');write_json(temporary,config);temporary.replace(active)
    print(json.dumps(config['routing_validation'],indent=2),flush=True)

if __name__=='__main__':main()

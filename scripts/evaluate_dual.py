"""Replay a verified validation cache, then report a fixed two-model candidate on test."""
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from foodquality8.consensus import ConsensusPredictor
from foodquality8.ai_detection.dual import DualPredictor,combine
from foodquality8.evaluation_cache import prepare_features
from scripts.evaluate_robust_consensus import summary
from scripts.check_cached_inference import equivalent

def strip(prediction):
 return combine([s for s in prediction['source_scores'] if s['source']!='SIC EfficientNet'],[e for e in prediction['quality_evidence'] if e['source']=='FoodQuality'],.8)

def metrics(rows):
 result=summary(rows);spoiled=[r for r in rows if r['condition']=='spoiled']
 result.update(spoiled_as_fresh=sum(r['prediction']['condition']=='fresh' for r in spoiled),spoiled_recall=sum(r['prediction']['condition']=='spoiled' and r['prediction']['food']==r['food'] for r in spoiled)/max(1,len(spoiled)))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 manifest=ROOT/'workspace/source9_v5/split.json';rows=json.loads(manifest.read_text(encoding='utf8'))
 active=json.loads((ROOT/'models/consensus.json').read_text(encoding='utf8'));original=ROOT/'model_runs/food9/expanded-v3/consensus.json';old=json.loads(original.read_text(encoding='utf8'))
 for key in ['detector_sha256','sic_classifier_sha256','quality_registry_sha256','calibration','quality_threshold']:
  assert active[key]==old[key],key
 cache=json.loads((ROOT/'model_runs/food9/expanded-v3/pipeline-validation-candidate.json').read_text(encoding='utf8'))
 assert cache['signature']['config_sha256']==hashlib.sha256(original.read_bytes()).hexdigest()
 assert cache['signature']['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
 config={k:v for k,v in active.items() if k not in ('sic_classifier','sic_classifier_sha256')};config.update(mode='two_models',quality_threshold=.8)
 dual_path=ROOT/'models/dual.json';dual_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
 validation={}
 for kind in ('clean','rotation','dim_blur','occlusion'):
  selected=[r for r in cache['records'] if r['kind']==kind]
  validation[kind]={'baseline':metrics(selected),'dual':metrics([dict(r,prediction=strip(r['prediction'])) for r in selected])}
 (a.output/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
 # Verify replay against the implementation on one image per food/condition.
 baseline=ConsensusPredictor(ROOT/'models/consensus.json');dual=DualPredictor(dual_path)
 features=prepare_features(baseline.base.data,rows,ROOT/'workspace/robust9_v3',ROOT/'workspace/source9_v5')
 selected={}
 for r in sorted(rows,key=lambda r:r['sha256']):
  if r['split']=='val':selected.setdefault((r['food'],r['condition']),r)
 for r in selected.values():
  with Image.open(r['path']) as im:
   b=baseline.predict_cached(im,*features[r['sha256']])[0];d=dual.predict_cached(im,*features[r['sha256']])[0]
   assert equivalent(strip(b),d),r['sha256']
 print('18 cached/direct pipeline parity checks passed',flush=True)
 test=[];started=time.perf_counter()
 for r in rows:
  if r['split']!='test':continue
  with Image.open(r['path']) as im:prediction=baseline.predict_cached(im,*features[r['sha256']])[0]
  test.append(dict(sha256=r['sha256'],food=r['food'],condition=r['condition'],baseline=prediction,dual=strip(prediction)))
  if len(test)%100==0:print('test',len(test),flush=True)
 report=dict(validation=validation,test={k:metrics([dict(r,prediction=r[k]) for r in test]) for k in ('baseline','dual')},parity_samples=len(selected),test_seconds=time.perf_counter()-started,threshold=.8,selection='Validation only; test used for reporting, not threshold selection.',limitation='Historical source test has been used in earlier project development; not a new independently collected external test.',records=test)
 (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report['test']),flush=True)
if __name__=='__main__':main()

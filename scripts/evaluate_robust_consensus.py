"""Compare deployed and candidate bundles on identical clean and perturbed validation images."""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
from foodquality8.consensus import ConsensusPredictor
from foodquality8.model_data import write_json


def perturb(image,kind):
    image=ImageOps.exif_transpose(image).convert('RGB')
    if kind=='rotation':
        return image.rotate(22,Image.Resampling.BICUBIC,expand=True,fillcolor=(124,116,104))
    if kind=='dim_blur':
        return ImageEnhance.Brightness(image).enhance(.78).filter(ImageFilter.GaussianBlur(.8))
    if kind=='occlusion':
        image=image.copy();w,h=image.size
        ImageDraw.Draw(image).rectangle((0,0,int(w*.2),int(h*.3)),fill=(124,116,104))
    return image


def summary(records):
    accepted=[r for r in records if r['prediction']['condition']!='unknown']
    correct=sum(r['prediction']['food']==r['food'] and r['prediction']['condition']==r['condition'] for r in accepted)
    return dict(samples=len(records),food_accuracy=sum(r['prediction']['food']==r['food'] for r in records)/max(1,len(records)),
                coverage=len(accepted)/max(1,len(records)),accepted_accuracy=correct/max(1,len(accepted)),
                correct_and_accepted=correct/max(1,len(records)))


def passes(baseline,candidate):
    clean=all(candidate['clean'][k]>=baseline['clean'][k] for k in ('food_accuracy','accepted_accuracy','correct_and_accepted'))
    robust=all(candidate[k]['food_accuracy']>=baseline[k]['food_accuracy'] for k in ('rotation','dim_blur','occlusion'))
    quality=all(candidate[k][metric]>=baseline[k][metric]
                for k in ('rotation','dim_blur') for metric in ('accepted_accuracy','correct_and_accepted'))
    return clean and robust and quality


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'models/consensus.json')
    p.add_argument('--manifest',type=Path,default=ROOT/'workspace/tomato_v3/split.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--full-clean',action='store_true')
    p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--features-base',type=Path)
    p.add_argument('--legacy-base',type=Path)
    args=p.parse_args()
    if bool(args.features_base)!=bool(args.legacy_base):raise ValueError('Provide both feature directories.')
    rows=[r for r in json.loads(args.manifest.read_text(encoding='utf8')) if r['split']==args.split and not r.get('derived_from')]
    from foodquality8.source_dataset import verify_files
    verify_files(rows)
    groups=defaultdict(list)
    for r in rows:groups[(r['food'],r['condition'])].append(r)
    subset=[]
    for key in sorted(groups):
        items=sorted(groups[key],key=lambda r:r['sha256']);random.Random(314).shuffle(items);subset+=items[:5]
    results={};hashes={};started=time.perf_counter()
    entries=[('baseline',args.baseline)]
    if not args.baseline_only:entries.append(('candidate',args.candidate))
    manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    for name,path in entries:
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        cache_path=args.output.with_name(args.output.stem+'-'+name+'.json')
        signature=dict(config_sha256=hashes[name],manifest_sha256=manifest_hash,
                       split=args.split,full_clean=args.full_clean,transforms_version=1,
                       cached_features=bool(args.features_base))
        if cache_path.exists():
            cached=json.loads(cache_path.read_text(encoding='utf8'))
            if cached.get('signature')==signature:
                results[name]=cached;print('Reused matching evaluation:',name,flush=True);continue
        model=ConsensusPredictor(path);records=[];metrics={}
        prepared={}
        if args.features_base:
            from foodquality8.evaluation_cache import prepare_features
            prepared=prepare_features(model.base.data,rows,args.features_base,args.legacy_base)
        for kind in ('clean','rotation','dim_blur','occlusion'):
            selected=rows if kind=='clean' and args.full_clean else subset
            section=[]
            for i,row in enumerate(selected):
                with Image.open(row['path']) as image:
                    if kind=='clean' and prepared:
                        prediction=model.predict_cached(image,*prepared[row['sha256']])[0]
                    else:
                        prediction=model.predict(perturb(image,kind))[0]
                section.append(dict(sha256=row['sha256'],food=row['food'],condition=row['condition'],kind=kind,prediction=prediction))
                if (i+1)%50==0:print(name,kind,i+1,'/',len(selected),flush=True)
            metrics[kind]=summary(section);records+=section
            if kind=='occlusion':
                # Occlusion can hide spoilage evidence; evaluate food identity only.
                metrics[kind]={k:v for k,v in metrics[kind].items() if k in ('samples','food_accuracy')}
        results[name]=dict(metrics=metrics,records=records,signature=signature)
        write_json(cache_path,results[name])
    if args.baseline_only:
        print(json.dumps(results['baseline']['metrics'],indent=2),flush=True);return
    report=dict(passed=passes(results['baseline']['metrics'],results['candidate']['metrics']),
                split=args.split,full_clean=args.full_clean,config_sha256=hashes,
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                metrics={k:v['metrics'] for k,v in results.items()},seconds=time.perf_counter()-started,
                limitation='Controlled 2D corruptions of internal holdouts, not unseen real camera views. Occlusion tests identity only.')
    write_json(args.output,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

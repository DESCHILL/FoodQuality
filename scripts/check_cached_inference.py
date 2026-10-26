"""Verify that cached features preserve complete three-source inference outputs."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from foodquality8.consensus import ConsensusPredictor
from foodquality8.evaluation_cache import prepare_features
from foodquality8.model_data import write_json
from foodquality8.source_dataset import verify_files


def equivalent(left,right):
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(equivalent(left[k],right[k]) for k in left)
    if isinstance(left,list):
        return len(left)==len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    if isinstance(left,float):return math.isclose(left,right,rel_tol=1e-4,abs_tol=1e-5)
    return left==right


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--features-base',type=Path,required=True)
    p.add_argument('--legacy-base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=json.loads(args.manifest.read_text(encoding='utf8'));selected={}
    for row in sorted(rows,key=lambda r:r['sha256']):
        if row['split']=='val':selected.setdefault((row['food'],row['condition']),row)
    rows=list(selected.values());verify_files(rows)
    model=ConsensusPredictor(args.registry)
    cached=prepare_features(model.base.data,rows,args.features_base,args.legacy_base)
    seconds={'normal':0.,'cached':0.};records=[]
    for row in rows:
        with Image.open(row['path']) as image:
            start=time.perf_counter();normal=model.predict(image);seconds['normal']+=time.perf_counter()-start
            start=time.perf_counter();fast=model.predict_cached(image,*cached[row['sha256']]);seconds['cached']+=time.perf_counter()-start
        record=dict(sha256=row['sha256'],food=row['food'],condition=row['condition'],equivalent=equivalent(normal,fast))
        records.append(record)
    passed=bool(records) and all(r['equivalent'] for r in records)
    report=dict(passed=passed,samples=len(records),seconds=seconds,records=records,
                registry_sha256=hashlib.sha256(args.registry.read_bytes()).hexdigest(),
                tolerance=dict(relative=1e-4,absolute=1e-5))
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
    if not passed:raise ValueError('Cached inference differs from normal inference.')


if __name__=='__main__':main()

"""Add source-labelled tomato photos with acquisition-date holdouts."""
import hashlib,json,random,re,shutil,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from foodquality8.core import phash
from foodquality8.food9 import encoder,visual_features
from foodquality8.model_data import write_json

def split_capture_dates(rows):
    days=sorted({r['capture_date'] for r in rows});random.Random(42).shuffle(days)
    n=max(1,round(len(days)*.2))
    mapping={d:'test' if i<n else 'val' if i<2*n else 'train' for i,d in enumerate(days)}
    for r in rows:r['split']=mapping[r['capture_date']]

def main():
    base=ROOT/'workspace/tomato_v3';base.mkdir(parents=True,exist_ok=True)
    if '--resplit' in sys.argv:
        rows=json.loads((base/'split.json').read_text(encoding='utf8'))
        split_capture_dates([r for r in rows if r.get('source')=='AgriFreshNET v1'])
        from foodquality8.source_dataset import validate_rows
        validate_rows(rows)
        write_json(base/'split.json',rows)
        report=json.loads((base/'data-report.json').read_text(encoding='utf8'))
        report['split_counts']=dict(Counter(r['split'] for r in rows));write_json(base/'data-report.json',report)
        features=np.load(base/'features.npz')['features']
        signature=hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in rows]).encode()).hexdigest()
        np.savez_compressed(base/'features.npz',features=features,signature=signature)
        print(report,flush=True);return
    old=json.loads((ROOT/'workspace/sic_merged/split.json').read_text(encoding='utf8'))
    buckets=defaultdict(set);hashes=[]
    def image_hash(path):
        with Image.open(path) as im:return int(phash(ImageOps.exif_transpose(im).convert('RGB')),16)
    def add(h):
        i=len(hashes);hashes.append(h)
        for j in range(5):buckets[(j,(h>>(13*j))&8191)].add(i)
    def overlaps(h):
        candidates=set()
        for j in range(5):candidates.update(buckets[(j,(h>>(13*j))&8191)])
        return any((h^hashes[i]).bit_count()<=4 for i in candidates)
    for r in old:add(image_hash(r['path']))
    source=ROOT/'food15_workspace/data/raw/agrifreshnet_v1/download/Processed Data'
    new=[];excluded=0
    for folder,condition in [('Fresh Tomato(1-10)','fresh'),('Rotten Tomato(24-35)','spoiled')]:
        for path in sorted((source/folder).glob('*.jpg')):
            h=image_hash(path)
            if overlaps(h):excluded+=1;continue
            date=re.search(r'IMG_(\d{8})_',path.name)
            if not date:continue
            add(h);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            dest=base/'images'/(digest+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            new.append(dict(path=str(dest),food='tomato',condition=condition,sha256=digest,
                duplicate_group='agrifreshnet_tomato_'+date[1],capture_date=date[1],source='AgriFreshNET v1',original=path.name))
    dates=defaultdict(list)
    for condition in ('fresh','spoiled'):
        days=sorted({r['capture_date'] for r in new if r['condition']==condition})
        random.Random(42).shuffle(days);n=max(1,round(len(days)*.2))
        if len(days)<5:raise ValueError('Insufficient independent acquisition dates.')
        mapping={d:'test' if i<n else 'val' if i<2*n else 'train' for i,d in enumerate(days)}
        for r in new:
            if r['condition']==condition:r['split']=mapping[r['capture_date']]
    split_capture_dates(new)
    merged=old+new;write_json(base/'split.json',merged)
    write_json(base/'data-report.json',dict(added=len(new),excluded_near_duplicates=excluded,split_counts=dict(Counter(r['split'] for r in merged)),
        new_class_counts=dict(Counter(r['condition'] for r in new)),
        source='https://data.mendeley.com/datasets/42m5tb7yv9/1',license='CC-BY-4.0',
        limitation='New images grouped by capture date; specimen identity unavailable. Public source labels, not independently reviewed. Semi-fresh excluded.'))
    model,transform=encoder();parts=[np.load(ROOT/'workspace/sic_merged/features.npz')['features']]
    with torch.inference_mode():
        for start in range(0,len(new),32):
            ims=[Image.open(r['path']).convert('RGB') for r in new[start:start+32]]
            parts.append(np.concatenate([model(torch.stack([transform(im) for im in ims])).numpy(),np.stack([visual_features(im) for im in ims])],axis=1))
            for im in ims:im.close()
            if start%160==0:print('Tomato features',start,'/',len(new),flush=True)
    signature=hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in merged]).encode()).hexdigest()
    np.savez_compressed(base/'features.npz',features=np.concatenate(parts),signature=signature)
    print((base/'data-report.json').read_text(),flush=True)

if __name__=='__main__':main()

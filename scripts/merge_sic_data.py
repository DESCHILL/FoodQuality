"""Import source-labelled SIC assets without modifying its checkout or old holdouts."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image, ImageOps
from foodquality8.core import phash
from foodquality8.food9 import FOODS
from foodquality8.model_data import write_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    args=parser.parse_args();repo=args.repo.resolve()
    out=ROOT/'workspace/sic_merged';out.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'models/sic';assets.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((repo/'HANDOVER_MANIFEST.json').read_text(encoding='utf8'))['files']
    provenance={}
    for relative,target in [('best.pt','detector.pt'),('models/efficientnet_37_fasttrack/efficientnet_b0_37_prototype.pth','condition.pth')]:
        source=repo/relative;digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=manifest[relative]['sha256']: raise ValueError('Checkpoint differs from handover: '+relative)
        shutil.copy2(source,assets/target)
        provenance[target]={'sha256':digest,'original':str(source)}
    write_json(assets/'provenance.json',{'repo':'https://github.com/manhhuu13112005/SIC-Food-Handover','files':provenance,
        'license_note':'Upstream detector model license not declared; local research/demo, not a redistribution grant.'})
    old=json.loads((ROOT/'food9_workspace/split.json').read_text(encoding='utf8'))
    original=json.loads((ROOT/'food9_workspace/catalog.json').read_text(encoding='utf8'))
    # Include even excluded original images: no source-neighbour may enter new train.
    hashes=[];buckets=defaultdict(set)
    def hash_image(path):
        with Image.open(path) as im: return int(phash(ImageOps.exif_transpose(im).convert('RGB')),16)
    def add(h):
        i=len(hashes);hashes.append(h)
        for part in range(5): buckets[(part,(h>>(13*part))&8191)].add(i)
    def overlaps(h):
        candidates=set()
        for part in range(5): candidates.update(buckets[(part,(h>>(13*part))&8191)])
        return any((h^hashes[i]).bit_count()<=4 for i in candidates)
    for row in original: add(hash_image(row['path']))
    records=list(csv.DictReader((repo/'datasets/external/agml_fruit_raw/manifest.csv').open(encoding='utf8')))
    counts=Counter();new=[];seen=set();archive=[]
    for row in records:
        src=repo/row['path'];raw=src.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=row['sha256']: raise ValueError('Source image changed: '+str(src))
        if digest in seen: counts['exact_duplicate']+=1;continue
        seen.add(digest)
        target=out/'images'/src.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target)
        food,state=row['target_class'].split('__')
        # Original source filenames explicitly distinguish Fresh vs Rotten;
        # do not reinterpret arbitrary visible_defect labels as rotten.
        name=row['source_image'].lower()
        condition='fresh' if name.startswith('fresh') else 'spoiled' if name.startswith('rotten') else None
        record=dict(path=str(target),food=food,condition=condition,sha256=digest,source='AgML raw / bdd69gyhv8.1',
            original=row['source_image'],source_label=row['target_class'],reviewed=False,license=row['license'])
        archive.append(record)
        if food not in FOODS or condition is None: counts['outside_training_scope']+=1;continue
        h=hash_image(target)
        if overlaps(h): counts['near_duplicate_excluded']+=1;continue
        # Index every accepted new image; conservative single representative.
        add(h);new.append(dict(record,duplicate_group=digest))
    by_class=defaultdict(list)
    for row in new: by_class[(row['food'],row['condition'])].append(row)
    for items in by_class.values():
        items.sort(key=lambda r:r['sha256']);random.Random(20261017).shuffle(items)
        n=max(1,round(len(items)*.2)) if len(items)>=5 else 0
        for i,row in enumerate(items): row['split']='test' if i<n else 'val' if i<2*n else 'train'
    merged=[dict(r,source='FoodClass V2') for r in old]+new
    write_json(out/'import-catalog.json',archive)
    write_json(out/'split.json',merged)
    write_json(out/'data-report.json',dict(original_images=len(old),imported_unique=len(archive),added=len(new),
        exclusions=dict(counts),new_by_class={f'{a}_{b}':len(v) for (a,b),v in by_class.items()},
        split_counts=dict(Counter(r['split'] for r in merged)),source_review='Source labels only; not manually approved.',
        limitation='Original holdouts preserved. Near-duplicates filtered against all FoodClass candidates. Original capture IDs and SIC training overlap unknown.'))
    shutil.copy2(repo/'datasets/external/agml_fruit_raw/source.json',out/'source.json')
    # Reuse unchanged ImageNet features, compute only new source images.
    import numpy as np
    from foodquality8.food9 import encoder,visual_features
    import torch
    oldcache=np.load(ROOT/'food9_workspace/features.npz')
    oldsignature=hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in old]).encode()).hexdigest()
    if str(oldcache['signature'])!=oldsignature:raise ValueError('Cached features do not match the original split.')
    oldfeatures=oldcache['features']
    model,transform=encoder();parts=[oldfeatures]
    with torch.inference_mode():
        for start in range(0,len(new),32):
            ims=[Image.open(r['path']).convert('RGB') for r in new[start:start+32]]
            parts.append(np.concatenate([model(torch.stack([transform(im) for im in ims])).numpy(),np.stack([visual_features(im) for im in ims])],axis=1))
            for im in ims: im.close()
            if start%160==0: print('New features',start,'/',len(new),flush=True)
    signature=hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in merged]).encode()).hexdigest()
    np.savez_compressed(out/'features.npz',features=np.concatenate(parts),signature=signature)
    print(json.dumps(json.loads((out/'data-report.json').read_text()),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

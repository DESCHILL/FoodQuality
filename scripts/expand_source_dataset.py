"""Expand fixed nine-food splits with dated source photos, preserving existing holdouts."""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image,ImageOps
from foodquality8.core import phash
from foodquality8.model_data import write_json
from foodquality8.source_dataset import validate_rows,verify_files

FOLDERS={
    'Fresh Banana(1-4)':('banana','fresh'), 'Rotten banana(7-13)':('banana','spoiled'),
    'Fresh Cucumber(1-6)':('cucumber','fresh'), 'Rotten Cucumber(12-20)':('cucumber','spoiled'),
    'Fresh Orange(1-9)':('orange','fresh'), 'Rotten Orange(20-35)':('orange','spoiled'),
}


def parent_name(name):
    return re.sub(r'^aug_\d+_','',name,flags=re.I).casefold()


def capture_date(name):
    match=re.search(r'IMG[_-](\d{8})',name,re.I)
    if not match:return None
    import datetime
    try:datetime.datetime.strptime(match[1],'%Y%m%d')
    except ValueError:return None
    return match[1]


def assign_date(date,existing):
    if date in existing:return existing[date]
    bucket=int(hashlib.sha256(('agrifreshnet:'+date).encode()).hexdigest()[:8],16)%100
    return ('test' if bucket<20 else 'val' if bucket<40 else 'train','agrifreshnet_date_'+date)


class HashIndex:
    def __init__(self):self.buckets=defaultdict(set);self.entries=[]
    def add(self,value,metadata):
        i=len(self.entries);self.entries.append((value,metadata))
        for part in range(5):self.buckets[(part,(value>>(13*part))&8191)].add(i)
    def matches(self,value):
        ids=set()
        for part in range(5):ids.update(self.buckets[(part,(value>>(13*part))&8191)])
        return [self.entries[i][1] for i in ids if (value^self.entries[i][0]).bit_count()<=4]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'food15_workspace/data/raw/agrifreshnet_v1/download/Processed Data')
    p.add_argument('--base',type=Path,default=ROOT/'workspace/tomato_v3')
    p.add_argument('--output',type=Path,default=ROOT/'workspace/source9_v4')
    args=p.parse_args()
    old=json.loads((args.base/'split.json').read_text(encoding='utf8'));validate_rows(old);verify_files(old)
    existing={}
    for r in old:
        if r.get('source')=='AgriFreshNET v1' and r.get('capture_date'):
            key=r['capture_date'];value=(r['split'],r['duplicate_group'])
            if key in existing and existing[key]!=value:raise ValueError('Existing capture date crosses splits.')
            existing[key]=value
    index=HashIndex();known={r['sha256']:r for r in old}
    for r in json.loads((ROOT/'food9_workspace/catalog.json').read_text(encoding='utf8')):
        known.setdefault(r['sha256'],r)
    for i,r in enumerate(known.values()):
        with Image.open(r['path']) as image:h=int(phash(ImageOps.exif_transpose(image).convert('RGB')),16)
        index.add(h,dict(food=r['food'],condition=r['condition'],new_index=None))
        if (i+1)%1000==0:print('Indexed source images',i+1,'/',len(known),flush=True)
    candidates=defaultdict(list);audit=Counter();excluded=[]
    for folder,(food,condition) in FOLDERS.items():
        directory=args.source/folder
        if not directory.is_dir():raise ValueError('Missing source folder: '+str(directory))
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):continue
            audit['source_files']+=1
            candidates[(food,parent_name(path.name))].append((condition,path))
    added=[];rejected=set()
    for (food,parent),items in sorted(candidates.items()):
        if len({c for c,_ in items})!=1:
            audit['conflicting_parent_labels']+=len(items);continue
        items.sort(key=lambda item:(bool(re.match(r'^aug_\d+_',item[1].name,re.I)),len(item[1].name),item[1].name))
        condition,path=items[0];audit['redundant_source_variants']+=len(items)-1
        date=capture_date(path.name)
        if date is None:
            audit['missing_capture_date']+=1;excluded.append(dict(file=path.name,reason='missing_capture_date'));continue
        try:
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            with Image.open(path) as image:
                image=ImageOps.exif_transpose(image).convert('RGB');w,h=image.size
                if min(w,h)<96:raise ValueError('Too small')
                hsh=int(phash(image),16)
        except (OSError,ValueError):audit['unreadable_or_small']+=1;continue
        matches=index.matches(hsh)
        if matches:
            audit['near_duplicate']+=1
            for match in matches:
                if (match['food'],match['condition'])!=(food,condition) and match['new_index'] is not None:
                    rejected.add(match['new_index']);audit['conflicting_near_duplicate']+=1
            continue
        split,group=assign_date(date,existing)
        destination=args.output/'images'/(digest+path.suffix.lower());destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(path,destination)
        row=dict(path=str(destination.resolve()),sha256=digest,food=food,condition=condition,split=split,
                 duplicate_group=group,capture_date=date,parent_capture=parent,
                 source='AgriFreshNET v1',original=path.name,source_folder=path.parent.name,
                 source_preprocessed=True,source_augmented=bool(re.match(r'^aug_\d+_',path.name,re.I)))
        index.add(hsh,dict(food=food,condition=condition,new_index=len(added)));added.append(row)
    added=[r for i,r in enumerate(added) if i not in rejected]
    merged=old+added;validate_rows(merged);verify_files(added)
    write_json(args.output/'split.json',merged)
    report=dict(previous_images=len(old),added_images=len(added),total_images=len(merged),audit=dict(audit),
                added_classes=dict(Counter(r['food']+'_'+r['condition'] for r in added)),
                added_split_classes=dict(Counter(r['split']+'/'+r['food']+'/'+r['condition'] for r in added)),
                split_counts=dict(Counter(r['split'] for r in merged)),
                added_capture_days=len({r['capture_date'] for r in added}),
                added_source_augmented=sum(r['source_augmented'] for r in added),
                limitations='Processed source; one representative per parent filename. Dates are group surrogates, not verified specimen IDs. Existing splits remain unchanged. No semi-fresh relabeling.')
    write_json(args.output/'data-report.json',report);write_json(args.output/'excluded.json',excluded)
    write_json(args.output/'sources.json',json.loads((ROOT/'config/robust9-sources.json').read_text(encoding='utf8')))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

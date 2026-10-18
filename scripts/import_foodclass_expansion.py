"""Import new source groups without moving or duplicating existing holdouts."""
import argparse
import hashlib
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image,ImageOps
from foodquality8.core import phash
from foodquality8.model_data import write_json
from foodquality8.source_dataset import validate_rows,verify_files
from scripts.expand_source_dataset import HashIndex


def select_groups(entries,hashes,reference_count):
    parents=list(range(len(entries)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    index=HashIndex()
    for i,value in enumerate(hashes):
        for j in index.matches(value):parents[find(i)]=find(j)
        index.add(value,i)
    groups=defaultdict(list)
    for i in range(len(entries)):groups[find(i)].append(i)
    selected=[];audit=Counter()
    for ids in groups.values():
        new=[i for i in ids if i>=reference_count]
        if not new:continue
        if any(i<reference_count for i in ids):
            audit['overlap_with_previous_sources']+=len(new);continue
        if len({(entries[i]['food'],entries[i]['condition']) for i in ids})!=1:
            audit['conflicting_source_labels']+=len(new);continue
        representative=min(new,key=lambda i:entries[i]['sha256'])
        row=dict(entries[representative]);digest=row['sha256']
        if 'path' in row:row['path']=str(Path(row['path']).resolve())
        bucket=int(hashlib.sha256(('foodclass-expansion:'+digest).encode()).hexdigest()[:8],16)%100
        row.update(split='test' if bucket<15 else 'val' if bucket<30 else 'train',
                   duplicate_group='foodclass_'+digest,source='FoodClass V2')
        selected.append(row);audit['redundant_new_images']+=len(new)-1
    return selected,dict(audit)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--previous-catalog',type=Path,default=ROOT/'food9_workspace/catalog.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    old=json.loads((args.base/'split.json').read_text(encoding='utf8'));validate_rows(old);verify_files(old)
    references={r['sha256']:r for r in old}
    for row in json.loads(args.previous_catalog.read_text(encoding='utf8')):references.setdefault(row['sha256'],row)
    incoming=json.loads(args.catalog.read_text(encoding='utf8'))
    entries=[];hashes=[];bad=0
    for row in [*references.values(),*incoming]:
        try:
            with Image.open(row['path']) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
                if min(image.size)<96:raise ValueError('Image too small')
                value=int(phash(image),16)
        except (OSError,ValueError):
            if row['sha256'] in references:raise
            bad+=1;continue
        entries.append(row);hashes.append(value)
        if len(entries)%2000==0:print('Kiểm tra nhóm ảnh',len(entries),flush=True)
    added,audit=select_groups(entries,hashes,len(references))
    merged=old+added;validate_rows(merged);verify_files(added)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'split.json',merged)
    report=dict(previous_images=len(old),downloaded_images=len(incoming),added_images=len(added),total_images=len(merged),
                unreadable_or_small=bad,audit=audit,split_counts=dict(Counter(r['split'] for r in merged)),
                added_classes=dict(Counter(r['food']+'_'+r['condition'] for r in added)),
                source_counts=dict(Counter(r.get('source','FoodClass V2') for r in merged)),
                limitation='One representative per transitive perceptual-hash group. Previous holdouts remain unchanged. Source labels and specimen independence are not independently verified.')
    write_json(args.output/'data-report.json',report)
    write_json(args.output/'sources.json',json.loads((ROOT/'config/robust9-sources.json').read_text(encoding='utf8')))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

"""Fetch a reproducible balanced subset using ZIP ranges; verify each member CRC."""
import concurrent.futures
import hashlib
import json
import struct
import time
import zlib
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]
FOODS=('Apple','Banana','Orange','Carrot','Cucumber','Tomato','Beef','Chicken','Mutton')

def main():
    import argparse,math
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'food9_workspace')
    parser.add_argument('--per-class',type=int,default=320)
    args=parser.parse_args()
    if args.per_class<40:raise ValueError('At least 40 images per class are required.')
    rows=json.loads((ROOT/'workspace/foodclass-index.json').read_text())
    source=json.loads((ROOT/'workspace/foodclass-files.json').read_text())[0]
    tasks=[]
    for food in FOODS:
        for condition in ('Fresh','Rotten'):
            members=sorted([r for r in rows if f'/{food}/{condition}/' in r['name'] and r['size']],key=lambda r:r['offset'])
            width=min(40,len(members))
            blocks=max(2,math.ceil(args.per_class/40))
            starts=range(0,len(members),40) if len(members)<=max(1000,args.per_class) else [round(k*(len(members)-width)/(blocks-1)) for k in range(blocks)]
            for start in starts:
                tasks.append((food.lower(),'fresh' if condition=='Fresh' else 'spoiled',members[start:start+width]))
    base=args.output.resolve()
    base.mkdir(parents=True,exist_ok=True)
    def fetch(task):
        food,condition,group=task
        folder=base/'images'/f'{food}_{condition}'
        folder.mkdir(parents=True,exist_ok=True)
        records=[]
        first=group[0]['offset']; last=group[-1]['offset']+30+len(group[-1]['name'].encode())+group[-1]['compressed']+1024
        expected=[folder/(hashlib.sha256(r['name'].encode()).hexdigest()[:24]+'.jpg') for r in group]
        blob=None
        if not all(p.exists() for p in expected):
            for attempt in range(4):
                try:
                    response=requests.get(source['content_details']['download_url'],headers={'Range':f'bytes={first}-{last}'},timeout=(30,120))
                    response.raise_for_status()
                    if response.status_code!=206:
                        raise RuntimeError('Server không trả đúng khoảng byte.')
                    blob=response.content
                    break
                except requests.RequestException:
                    if attempt==3: raise
                    time.sleep(2*(attempt+1))
        for member,path in zip(group,expected):
            if not path.exists():
                off=member['offset']-first
                header=struct.unpack_from('<4s5H3L2H',blob,off)
                if header[0]!=b'PK\x03\x04': raise ValueError('Sai ZIP header')
                begin=off+30+header[-2]+header[-1]
                compressed=blob[begin:begin+member['compressed']]
                raw=zlib.decompress(compressed,-15) if member['method']==8 else compressed
                if len(raw)!=member['size'] or zlib.crc32(raw)&0xffffffff!=member['crc']:
                    raise ValueError('CRC ảnh không khớp: '+member['name'])
                path.write_bytes(raw)
            raw=path.read_bytes()
            if zlib.crc32(raw)&0xffffffff!=member['crc']: raise ValueError('Ảnh lưu trên máy đã đổi.')
            records.append(dict(path=str(path),food=food,condition=condition,original=member['name'],sha256=hashlib.sha256(raw).hexdigest()))
        return records
    result={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,records in enumerate(pool.map(fetch,tasks),1):
            for row in records: result[row['original']]=row
            if i%8==0: print(f'Tải nhóm {i}/{len(tasks)}; đã xác minh {len(result)} ảnh',flush=True)
    (base/'catalog.json').write_text(json.dumps(list(result.values()),ensure_ascii=False,indent=2),encoding='utf8')
    (base/'source.json').write_text(json.dumps(dict(name='FoodClass V2',doi='10.17632/5fjkkv5xgt.2',
        url='https://data.mendeley.com/datasets/5fjkkv5xgt/2',license='CC-BY-4.0',
        attribution='Musarat Khan, Farhana Afrin, Afia Ebnath (2026). FoodClass. Mendeley Data V2.',
        sampling=f'All images for classes <=max(1000,{args.per_class}); otherwise evenly spaced blocks of 40 toward target {args.per_class}; subset, not entire archive',
        integrity='ZIP CRC checked for every downloaded image; SHA256 saved per image; full archive not downloaded',
        images=len(result)),ensure_ascii=False,indent=2),encoding='utf8')
    print('Hoàn tất bộ ảnh 9 loại:',len(result),flush=True)

if __name__=='__main__': main()

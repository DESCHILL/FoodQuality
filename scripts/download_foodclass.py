"""Download the author's public CC BY archive, verify its published SHA256."""
import hashlib
import json
import sys
from pathlib import Path

import requests

root=Path(__file__).resolve().parents[1]
entry=json.loads((root/'workspace/foodclass-files.json').read_text(encoding='utf8'))[0]
target=root/'food9_workspace/source/FoodClass2.zip'
target.parent.mkdir(parents=True,exist_ok=True)
partial=target.with_suffix('.partial')
expected=entry['content_details']['sha256_hash']
if not target.exists():
    start=partial.stat().st_size if partial.exists() else 0
    response=requests.get(entry['content_details']['download_url'],headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(30,120))
    response.raise_for_status()
    if start and response.status_code!=206:
        raise RuntimeError('Server không hỗ trợ tiếp tục tải; giữ file tải dở.')
    count=start
    progress=-1
    with partial.open('ab' if start else 'wb') as output:
        for chunk in response.iter_content(4*1024*1024):
            output.write(chunk)
            count+=len(chunk)
            percent=int(100*count/entry['size'])
            if percent//5!=progress:
                print(f'Tải FoodClass: {percent}% ({count//1024//1024} MB)',flush=True)
                progress=percent//5
    if count!=entry['size']:
        raise RuntimeError('Tệp chưa tải đủ; chạy lại để tiếp tục.')
    path=partial
else:
    path=target
with path.open('rb') as stream:
    actual=hashlib.file_digest(stream,'sha256').hexdigest()
if actual!=expected:
    raise RuntimeError('SHA256 không khớp dữ liệu tác giả.')
if path==partial:
    partial.replace(target)
manifest=dict(name='FoodClass',version=2,doi='10.17632/5fjkkv5xgt.2',
    source='https://data.mendeley.com/datasets/5fjkkv5xgt/2',license='CC-BY-4.0',
    attribution='Musarat Khan, Farhana Afrin, Afia Ebnath (2026), FoodClass, Mendeley Data V2.',
    sha256=actual,bytes=target.stat().st_size,download=entry['content_details']['download_url'])
(target.parent/'source.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('Đã tải và xác minh SHA256:',target,flush=True)

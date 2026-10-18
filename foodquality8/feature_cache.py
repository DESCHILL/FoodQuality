"""Resume frozen-encoder feature extraction and reuse only matching image recipes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps
from .robust_views import LEGACY,feature_signature,extract_features


def signatures(rows,recipe):
    values={feature_signature(rows,recipe)}
    if recipe==LEGACY:
        values.add(hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in rows]).encode()).hexdigest())
    return values


def atomic_cache(path,**values):
    path=Path(path);temporary=path.with_suffix('.pending.npz')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
    temporary.replace(path)


def ensure_features(base,rows,model,recipe,reuse=(),log=print):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    cache=base/'features.npz';signature=feature_signature(rows,recipe)
    if cache.exists():
        with np.load(cache) as stored:
            if str(stored['signature']) in signatures(rows,recipe):
                values=stored['features']
                if values.shape!=(len(rows),1331) or not np.isfinite(values).all():raise ValueError('Invalid feature cache.')
                return values
    values=np.zeros((len(rows),1331),dtype=np.float32);ready=np.zeros(len(rows),dtype=bool)
    progress=base/'features.partial.npz'
    if progress.exists():
        with np.load(progress) as stored:
            if str(stored['signature'])==signature:
                values=stored['features'];ready=stored['ready']
                if values.shape!=(len(rows),1331) or ready.shape!=(len(rows),) or not np.isfinite(values[ready]).all():
                    raise ValueError('Invalid partial feature cache.')
    lookup={}
    for i,row in enumerate(rows):lookup.setdefault(row['sha256'],[]).append(i)
    for directory in reuse:
        directory=Path(directory)
        source_rows=json.loads((directory/'split.json').read_text(encoding='utf8'))
        with np.load(directory/'features.npz') as stored:
            if str(stored['signature']) not in signatures(source_rows,recipe):
                raise ValueError('Reuse cache has a different recipe or manifest: '+str(directory))
            source=stored['features']
            if source.shape!=(len(source_rows),1331) or not np.isfinite(source).all():raise ValueError('Invalid reuse features.')
            for j,row in enumerate(source_rows):
                for i in lookup.get(row['sha256'],[]):
                    if not ready[i]:values[i]=source[j];ready[i]=True
    missing=np.flatnonzero(~ready);log(f'Đặc trưng có sẵn: {int(ready.sum())}/{len(rows)}; cần xử lý thêm {len(missing)} ảnh.')
    batch=8 if recipe['views']==2 else 32
    for offset in range(0,len(missing),batch):
        ids=missing[offset:offset+batch];images=[]
        for i in ids:
            with Image.open(rows[i]['path']) as image:images.append(ImageOps.exif_transpose(image).convert('RGB'))
        values[ids]=extract_features(model,images,recipe).numpy();ready[ids]=True
        if offset%320==0:
            atomic_cache(progress,features=values,ready=ready,signature=signature)
            log(f'Trích thêm {min(offset+batch,len(missing))}/{len(missing)} ảnh.')
    if not np.isfinite(values).all():raise ValueError('Non-finite extracted features.')
    atomic_cache(cache,features=values,signature=signature)
    if progress.exists():progress.unlink()
    return values

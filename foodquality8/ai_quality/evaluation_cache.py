"""Align verified frozen-encoder caches to evaluation images by content hash."""
import json
from pathlib import Path

import numpy as np
import torch
from foodquality8.ai_quality.feature_cache import signatures
from foodquality8.ai_quality.robust_views import LEGACY


def align_features(base,recipe,rows):
    base=Path(base)
    stored_rows=json.loads((base/'split.json').read_text(encoding='utf8'))
    with np.load(base/'features.npz') as cache:
        if str(cache['signature']) not in signatures(stored_rows,recipe):
            raise ValueError('Evaluation feature recipe or manifest mismatch.')
        features=cache['features']
    if features.shape!=(len(stored_rows),1331) or not np.isfinite(features).all():
        raise ValueError('Invalid evaluation feature matrix.')
    lookup={r['sha256']:i for i,r in enumerate(stored_rows)}
    if any(r['sha256'] not in lookup for r in rows):raise ValueError('Missing evaluation image features.')
    return features[[lookup[r['sha256']] for r in rows]]


def prepare_features(checkpoint,rows,detail_base,legacy_base):
    from foodquality8.ai_quality.food9 import encoder
    standard,_=encoder()
    reference=standard.state_dict()
    if set(checkpoint['encoder'])!=set(reference) or not all(torch.equal(v,reference[k]) for k,v in checkpoint['encoder'].items()):
        raise ValueError('Feature reuse requires the original frozen ImageNet encoder.')
    legacy=align_features(legacy_base,LEGACY,rows)
    recipe=checkpoint.get('preprocessing',LEGACY)
    detail=legacy if recipe==LEGACY else align_features(detail_base,recipe,rows)
    return {r['sha256']:(detail[i],legacy[i]) for i,r in enumerate(rows)}

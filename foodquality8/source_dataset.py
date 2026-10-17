"""Validate group separation before using prepared source-labelled training data."""
from .food9 import FOODS

def validate_rows(rows):
    groups={};hashes={};counts={}
    for row in rows:
        split=row['split'];label=(row['food'],row['condition'])
        if split not in ('train','val','test') or label[0] not in FOODS or label[1] not in ('fresh','spoiled'):
            raise ValueError('Nhãn hoặc tập dữ liệu không hợp lệ.')
        group=row['duplicate_group'];digest=row['sha256']
        if group in groups and groups[group]!=split:
            raise ValueError('Ảnh trùng giữa các tập hoặc xung đột nhãn.')
        if digest in hashes and hashes[digest]!=(split,label):
            raise ValueError('Ảnh trùng giữa các tập hoặc xung đột nhãn.')
        groups[group]=split;hashes[digest]=(split,label)
        counts[(split,label)]=counts.get((split,label),0)+1
    for split in ('train','val','test'):
        for food in FOODS:
            for condition in ('fresh','spoiled'):
                if (split,(food,condition)) not in counts:
                    raise ValueError(f'Thiếu lớp {food}/{condition} trong {split}.')
    return counts

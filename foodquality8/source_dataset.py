"""Validate group separation before using prepared source-labelled training data."""
from .food9 import FOODS


def verify_files(rows):
    import hashlib
    from pathlib import Path
    for row in rows:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Nội dung ảnh khác manifest: '+row['path'])

def validate_rows(rows):
    groups={};hashes={};counts={}
    originals={r['sha256']:r for r in rows if not r.get('derived_from')}
    for row in rows:
        split=row['split'];label=(row['food'],row['condition'])
        if row.get('derived_from'):
            parent=originals.get(row['derived_from'])
            if (parent is None or split!='train' or parent['split']!='train'
                or any(row[k]!=parent[k] for k in ('food','condition','duplicate_group'))):
                raise ValueError('Ảnh biến đổi phải giữ nhãn, nhóm và tập train của ảnh gốc.')
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

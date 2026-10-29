"""Validate immutable toolkit releases before training either model."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


def check_release(root, kind, experimental=False):
    root = Path(root).resolve()
    if kind not in ('classifier', 'detector'):
        raise ValueError('Unknown model type')
    manifest_path = root / 'manifest.json'
    metadata_path = root / 'metadata.csv'
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise ValueError('Chon thu muc release da xuat (co manifest.json va metadata.csv). Chua co release: duyet, chia tap va xuat dataset truoc.')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    auto=manifest.get('label_mode')=='automatic_suggested'
    suggested=manifest.get('label_mode')=='google_query_suggested' or auto
    if suggested and (not experimental or (kind!='classifier' and not auto)):
        raise ValueError('Dữ liệu nhãn gợi ý chỉ dùng cho huấn luyện phân loại thử nghiệm, không có box YOLO.')
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != manifest.get('catalog_sha256'):
        raise ValueError('Metadata release da bi thay doi.')
    with metadata_path.open(encoding='utf-8-sig', newline='') as stream:
        metadata = list(csv.DictReader(stream))
    sources = json.loads((root/'sources.json').read_text(encoding='utf8'))
    groups, images = {}, {}
    labels = {}
    for row in metadata:
        if not suggested and sources.get(row['source_dataset'], {}).get('license_status') != 'Approved':
            raise ValueError('Nguon anh chua duoc duyet quyen su dung.')
        valid_status=row['review_status']=='Approved' or (suggested and row['review_status']=='Suggested' and row.get('label_origin') in ('google_query_suggested','local_metadata_suggested'))
        if not valid_status or row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Release co anh chua duyet hoac chua chia tap.')
        if row['image_id'] in images:
            raise ValueError('Trung image_id trong release.')
        images[row['image_id']] = row
        for field in ('group_id', 'capture_group', 'pixel_hash', 'sha256'):
            value = row.get(field)
            if value:
                key = (field, value)
                if key in groups and groups[key] != row['split']:
                    raise ValueError('Ro ri du lieu giua train/val/test.')
                groups[key] = row['split']
        label = row['food_name'] + '_' + row['condition']
        labels[label] = dict(food=row['food_name'], condition=row['condition'])
    if kind == 'classifier':
        classes = sorted(manifest.get('classes', []))
        if len(classes) < 2 or any(c not in labels for c in classes):
            raise ValueError('Classifier can it nhat 2 lop co nhan va du anh trong 3 tap.')
        base = root / 'classification'
        names = classes
    else:
        import yaml
        data = yaml.safe_load((root / 'detection/data.yaml').read_text(encoding='utf8'))
        names = data['names']
        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError('YOLO class names khong hop le.')
        base = root / 'detection/images'
        classes = []
    counts = {}
    pixels = {}
    release_hash = hashlib.sha256(metadata_path.read_bytes())
    for split in ('train', 'val', 'test'):
        split_counts = Counter()
        files = sorted((base / split).rglob('*.jpg'))
        if not files:
            raise ValueError(f'{kind}: tap {split} rong. YOLO can bounding box da duyet; classifier can nhan chat luong.')
        for path in files:
            row = images.get(path.stem)
            if not row or row['split'] != split:
                raise ValueError('Anh khong khop metadata: ' + str(path))
            with Image.open(path) as source:
                im = ImageOps.exif_transpose(source).convert('RGB')
                digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            if digest in pixels and pixels[digest] != split:
                raise ValueError('Anh trung pixel giua cac tap.')
            pixels[digest] = split
            release_hash.update(path.relative_to(root).as_posix().encode())
            release_hash.update(path.read_bytes())
            if kind == 'classifier':
                label = row['food_name'] + '_' + row['condition']
                if path.parent.name != label or label not in classes:
                    raise ValueError('Thu muc lop khong khop metadata.')
                split_counts[label] += 1
            else:
                if row['bbox_status'] != 'Approved' and not (auto and row['bbox_status']=='Suggested' and row.get('bbox_origin')=='grabcut_foreground_suggestion'):
                    raise ValueError('Bounding box chua duyet.')
                annotation = root / 'detection/labels' / split / (path.stem + '.txt')
                text = annotation.read_text(encoding='utf8')
                if not text.strip():
                    raise ValueError('Thieu bounding box: ' + str(annotation))
                release_hash.update(text.encode())
                for line in text.splitlines():
                    values = line.split()
                    if len(values) != 5:
                        raise ValueError('Nhan YOLO phai co class x y w h.')
                    c, x, y, w, h = map(float, values)
                    if not (c.is_integer() and 0 <= c < len(names) and 0 < w <= 1 and 0 < h <= 1
                            and 0 <= x-w/2 <= x+w/2 <= 1 and 0 <= y-h/2 <= y+h/2 <= 1):
                        raise ValueError('Bounding box YOLO ngoai anh / class sai.')
                    split_counts[names[int(c)]] += 1
        expected = {r['image_id'] for r in metadata if r['split']==split and
                    (kind=='classifier' or ((r['bbox_status']=='Approved' or (auto and r['bbox_status']=='Suggested')) and json.loads(r.get('bbox_json') or '[]')))}
        if {p.stem for p in files} != expected:
            raise ValueError(f'{split}: tep anh khong day du theo metadata release.')
        missing = set(names) - set(split_counts)
        if missing:
            raise ValueError(f'{split} thieu lop: {", ".join(sorted(missing))}')
        counts[split] = dict(split_counts)
    return dict(root=str(root), kind=kind, counts=counts, classes=names,label_mode=manifest.get('label_mode','reviewed'),
                class_labels=[labels[c] for c in classes], dataset_sha256=release_hash.hexdigest())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)

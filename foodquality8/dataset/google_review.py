"""Bridge downloaded Google candidates into the existing reviewed dataset catalog."""
import hashlib
import json
import shutil
import tempfile
import unicodedata
from pathlib import Path

FOOD_ENGLISH = {
    'thit bo': 'beef', 'bo': 'beef', 'thit lon': 'pork', 'thit heo': 'pork',
    'lon': 'pork', 'heo': 'pork', 'thit ga': 'chicken', 'ga': 'chicken',
    'tao': 'apple', 'chuoi': 'banana', 'cam': 'orange', 'xoai': 'mango',
    'dau tay': 'strawberry', 'nho': 'grape', 'oi': 'guava', 'ca chua': 'tomato',
    'khoai tay': 'potato', 'dua chuot': 'cucumber', 'dua leo': 'cucumber',
    'ca rot': 'carrot', 'ot chuong': 'bell_pepper',
}


def normalized_food(name, foods):
    name = name.lower().strip().replace('đ', 'd')
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = FOOD_ENGLISH.get(name, name).replace('chicken meat', 'chicken').replace(' ', '_')
    return name if name in foods else 'unknown'


def import_google(dataset, downloads):
    downloads = Path(downloads).resolve()
    known = dataset.sources()
    added = 0
    errors = []
    for manifest in sorted(downloads.glob('*/manifest.jsonl')):
        for line_number, line in enumerate(manifest.read_text(encoding='utf8').splitlines(), 1):
            try:
                item = json.loads(line)
                image = Path(item['img_path']).resolve()
                if not image.is_relative_to(downloads):
                    raise ValueError('Image path outside downloads folder')
                if hashlib.sha256(image.read_bytes()).hexdigest() != item['sha256']:
                    raise ValueError('Image checksum changed')
                sid = 'google_' + hashlib.sha256((item.get('source_url', '') + item['sha256']).encode()).hexdigest()[:24]
                if sid in known and any(r['source_dataset'] == sid for r in dataset.rows()):
                    continue
                food = normalized_food(item.get('food_name', ''), dataset.foods)
                source = known.get(sid) or dict(
                    id=sid, source_url=item.get('source_url', ''),
                    google_url=item.get('google_url', ''), search_query=item.get('query', ''),
                    search_stage=item.get('search_stage', ''), downloaded_at=item.get('downloaded_at', ''),
                    license_status='Pending', license_id='', license_url='', checked_at='',
                    redistribution_allowed=False,
                    mapping={food: [food, 'unknown']} if food != 'unknown' else {})
                with tempfile.TemporaryDirectory(dir=dataset.root) as temp:
                    temp = Path(temp)
                    cfg = temp / 'source.json'
                    cfg.write_text(json.dumps({'sources': [source]}), encoding='utf8')
                    dataset.configure(cfg)
                    image_dir = temp / 'images' / food
                    image_dir.mkdir(parents=True)
                    shutil.copyfile(image, image_dir / (item['sha256'] + image.suffix))
                    added += dataset.ingest(sid, temp / 'images')
                known[sid] = source
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f'{manifest.name}:{line_number}: {exc}')
    dataset.coverage()
    return dict(added=added, errors=errors)


def save_review(dataset, image_id, food, condition, status, reviewer='', evidence='', capture_group=''):
    current = next((r for r in dataset.rows() if r['image_id'] == image_id), None)
    if current is None:
        raise ValueError('Khong tim thay anh.')
    if status not in ('Pending', 'Approved', 'Rejected'):
        raise ValueError('Trang thai khong hop le.')
    if not dataset.valid_label(food, condition) and not (food in [*dataset.foods, 'unknown'] and condition == 'unknown'):
        raise ValueError('Nhan khong thuoc bo nhan cua project.')
    if status == 'Approved':
        if current['error'] or not dataset.valid_label(food, condition):
            raise ValueError('Khong the duyet anh loi hoac nhan chua xac dinh.')
        if not reviewer.strip() or not evidence.strip() or not capture_group.strip():
            raise ValueError('Can ten nguoi duyet, can cu gan nhan va nhom mau/phien chup.')
        path = dataset.root / current['raw_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != current['sha256']:
            raise ValueError('Anh da thay doi sau khi nhap; khong the duyet.')
    with dataset.db:
        if food != current['food_name']:
            dataset.db.execute("UPDATE images SET bbox_status='Pending' WHERE image_id=?", (image_id,))
        dataset.db.execute('UPDATE images SET food_name=?,category=?,condition=?,review_status=?,reviewer=?,label_evidence=?,capture_group=? WHERE image_id=?',
                           (food, dataset.foods.get(food, 'unknown'), condition, status, reviewer.strip(), evidence.strip(), capture_group.strip(), image_id))
        dataset.db.execute("UPDATE images SET split=''")
        dataset.audit('gui_review', dict(image_id=image_id, food=food, condition=condition, status=status, reviewer=reviewer, evidence=evidence, capture_group=capture_group))
    dataset.deduplicate()
    dataset.snapshot()
    dataset.coverage()

"""Build one dataset index without changing labels, splits or provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def consolidate(manifests, destination):
    destination = Path(destination).resolve()
    records, seen, groups = [], {}, {}
    origins = Counter()
    inputs = []
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        raw = manifest.read_bytes()
        inputs.append(dict(path=str(manifest), sha256=hashlib.sha256(raw).hexdigest()))
        for original in json.loads(raw):
            row = dict(original)
            path = Path(row['path'])
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                raise ValueError(f'Missing image: {path}')
            row['path'] = str(path.resolve())
            digest = row['sha256']
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Image hash mismatch: {path}')
            if row['split'] not in ('train', 'val', 'test'):
                raise ValueError(f'Invalid split: {path}')
            if digest in seen:
                if seen[digest] != row:
                    raise ValueError(f'Conflicting record or provenance: {digest}')
                continue
            group = row.get('duplicate_group', digest)
            if group in groups and groups[group] != row['split']:
                raise ValueError(f'Group crosses train/val/test: {group}')
            groups[group] = row['split']
            seen[digest] = row
            records.append(row)
            origins[row.get('source', 'unspecified')] += 1
    if not records:
        raise ValueError('Dataset is empty')
    if destination.exists():
        raise FileExistsError(f'Choose a new dataset directory: {destination}')
    destination.mkdir(parents=True)
    (destination / 'split.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
    metadata = dict(name='FoodQuality', images=len(records), storage='original_files',
                    splits=dict(Counter(r['split'] for r in records)),
                    provenance=dict(origins), inputs=inputs)
    (destination / 'dataset.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Gộp chỉ mục dữ liệu FoodQuality, giữ nguyên nguồn gốc và tập đánh giá.')
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.manifest, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

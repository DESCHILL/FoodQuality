"""Derive reproducible training images without changing source labels or holdouts."""
import argparse
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from foodquality8.model_data import write_json
from foodquality8.source_dataset import validate_rows,verify_files


def augment(image, seed):
    rng = random.Random(seed)
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    if rng.random() < .5:
        image = ImageOps.mirror(image)
    # Expansion preserves the complete object, including small spoilage cues.
    image = image.rotate(rng.choice([-12, -8, 8, 12]), Image.Resampling.BICUBIC,
                         expand=True, fillcolor=(124, 116, 104))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.9, 1.1))
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .45)))
    return image


def build(rows, output):
    validate_rows(rows)
    verify_files(rows)
    if any(r.get('derived_from') for r in rows):
        raise ValueError('Use original source rows, not an already augmented dataset.')
    output = Path(output).resolve()
    directory = output / 'images'
    directory.mkdir(parents=True, exist_ok=True)
    previous={}
    if (output/'split.json').exists():
        previous={r['derived_from']:r for r in json.loads((output/'split.json').read_text(encoding='utf8'))
                  if r.get('augmentation')=='rotation-light-blur-v1' and r.get('derived_from')}
    result = [dict(r) for r in rows]
    for row in rows:
        if row['split'] != 'train':
            continue
        seed = int(row['sha256'][:16], 16)
        cached=previous.get(row['sha256'])
        if cached and Path(cached['path']).is_file() and hashlib.sha256(Path(cached['path']).read_bytes()).hexdigest()==cached['sha256']:
            result.append(dict(row,path=cached['path'],sha256=cached['sha256'],derived_from=row['sha256'],
                               augmentation='rotation-light-blur-v1',augmentation_seed=seed))
            continue
        with Image.open(row['path']) as source:
            image = augment(source, seed)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95, subsampling=0)
        raw = buffer.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        path = directory / (digest + '.jpg')
        if not path.exists():
            path.write_bytes(raw)
        result.append(dict(row, path=str(path), sha256=digest,
                           derived_from=row['sha256'], augmentation='rotation-light-blur-v1',
                           augmentation_seed=seed))
    validate_rows(result)
    write_json(output/'split.json', result)
    report = dict(original_images=len(rows), derived_images=len(result)-len(rows),
                  total_images=len(result),
                  independent_training_groups=len({r['duplicate_group'] for r in rows if r['split']=='train'}),
                  split_counts=dict(Counter(r['split'] for r in result)),
                  class_counts=dict(Counter(r['food']+'_'+r['condition'] for r in result)),
                  source_counts=dict(Counter(r.get('source','FoodClass V2') for r in rows)),
                  limitation='Derived images are not independent samples. Validation/test originals are unchanged. '
                             '2D rotation cannot synthesize unseen surfaces or verify freshness labels.')
    write_json(output/'data-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'workspace/tomato_v3/split.json')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/robust9_v1')
    args = parser.parse_args()
    print(json.dumps(build(json.loads(args.source.read_text(encoding='utf8')), args.output), indent=2))


if __name__ == '__main__':
    main()

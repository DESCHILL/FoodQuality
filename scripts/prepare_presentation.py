"""Prepare synthetic input only. Import, review, split and export remain manual."""
import json
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    root = Path('presentation_workspace')
    target = root / 'input'
    if target.exists():
        raise SystemExit('Input already exists. Continue with configure/ingest; existing files were preserved.')
    rng = np.random.default_rng(2026)
    for label in ('FreshApple', 'RottenApple'):
        folder = target / label
        folder.mkdir(parents=True)
        for i in range(20):
            Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)).save(folder / f'{i:02}.png')
    (target / 'FreshApple/copy.png').write_bytes((target / 'FreshApple/00.png').read_bytes())
    (target / 'FreshApple/broken.jpg').write_bytes(b'This is not an image')
    config = {'sources': [dict(id='presentation', source_url='local:synthetic-fixtures',
        owner='Local demo generator', license_id='OWN', license_url='local:generated-fixtures',
        checked_at=date.today().isoformat(), license_status='Approved', redistribution_allowed=True,
        mapping={'FreshApple': ['apple', 'fresh'], 'RottenApple': ['apple', 'visibly_spoiled']})]}
    (root / 'sources.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Prepared 40 synthetic images, 1 exact copy and 1 invalid image.')
    print('No images were imported or approved. Continue with the manual steps.')


if __name__ == '__main__':
    main()

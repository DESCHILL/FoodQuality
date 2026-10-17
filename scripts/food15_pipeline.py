"""Run from VS Code (F5), or python -m scripts.food15_pipeline.

Uses only local deterministic code. Does not call an AI or approve labels.
"""
import json
from collections import Counter
from pathlib import Path

from foodquality8 import Dataset


def main():
    project = Path(__file__).resolve().parents[1]
    root = project / 'food15_workspace'
    dataset = Dataset(root)
    try:
        dataset.set_taxonomy(project / 'config/food15-taxonomy.json')
        dataset.configure(project / 'config/food15-sources.json')
        for sid, source in dataset.sources().items():
            directory = root / 'data/raw' / sid / 'download'
            if not directory.exists():
                print(f'{sid}: source download not present; no images fabricated.', flush=True)
                continue
            if not source.get('mapping'):
                print(f'{sid}: source folders need an explicit label mapping.', flush=True)
                continue
            print(f'Inspecting real source images: {sid}', flush=True)
            added = dataset.ingest(sid, directory)
            print(f'{sid}: added {added} records', flush=True)
        print('Writing review workbook...', flush=True)
        workbook = dataset.review_export(root / 'reports/FoodQuality-15-5-review.xlsx')
        summary = dataset.coverage()
        print('Original labels:', dict(Counter(r['original_label'] for r in dataset.rows())), flush=True)
        print('Foods with candidate images:', dict(Counter(r['food_name'] for r in dataset.rows())), flush=True)
        print('Review workbook:', workbook, flush=True)
        print(json.dumps({k:v for k,v in summary.items() if k != 'missing_classes'}, indent=2), flush=True)
        print(f"Training release requires reviewed data for the {summary['target_classes']} configured labels; search keywords are not ground truth.", flush=True)
        (root / 'reports/pipeline-last-run.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from foodquality8 import Dataset
from foodquality8.google_review import import_google, save_review


class GoogleReviewTests(unittest.TestCase):
    def test_import_review_integrity_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            downloads = root / 'downloads' / 'thit_bo'
            downloads.mkdir(parents=True)
            image = downloads / 'one.jpg'
            Image.new('RGB', (300, 300), 'brown').save(image)
            sha = hashlib.sha256(image.read_bytes()).hexdigest()
            (downloads / 'manifest.jsonl').write_text(json.dumps(dict(
                img_path=str(image), sha256=sha, food_name='thịt bò',
                query='thịt bò tươi', source_url='https://example.org/one.jpg')), encoding='utf8')
            taxonomy = root / 'taxonomy.json'
            taxonomy.write_text(json.dumps({'foods': {'beef': 'meat'}, 'conditions': ['fresh', 'spoiled']}))
            d = Dataset(root / 'dataset')
            try:
                d.set_taxonomy(taxonomy)
                self.assertEqual(import_google(d, root / 'downloads')['added'], 1)
                self.assertEqual(import_google(d, root / 'downloads')['added'], 0)
                row = d.rows()[0]
                self.assertEqual(row['food_name'], 'beef')
                self.assertEqual(row['condition'], 'unknown')
                with self.assertRaises(ValueError):
                    save_review(d, row['image_id'], 'beef', 'fresh', 'Approved')
                save_review(d, row['image_id'], 'beef', 'fresh', 'Approved', 'tester', 'fixture only', 'sample1')
                self.assertFalse(d.eligible(d.rows()[0]))  # Label review cannot approve source rights.
                import_google(d, root / 'downloads')
                self.assertEqual(d.rows()[0]['review_status'], 'Approved')
                (d.root / row['raw_path']).write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    save_review(d, row['image_id'], 'beef', 'fresh', 'Approved', 'tester', 'evidence', 'sample1')
            finally:
                d.close()


if __name__ == '__main__':
    unittest.main()

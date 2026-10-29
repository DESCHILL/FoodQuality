import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from foodquality8 import module_path
from foodquality8.dataset.unified import consolidate


class ModuleStructureTests(unittest.TestCase):
    def test_legacy_modules_alias_canonical_implementations(self):
        for name, role in [('training', 'ai_quality'), ('consensus', 'ai_detection'),
                           ('core', 'dataset'), ('model_data', 'dataset')]:
            self.assertIs(importlib.import_module('foodquality8.' + name),
                          importlib.import_module(f'foodquality8.{role}.{name}'))
            self.assertEqual(module_path(name).parent.name, role)
        self.assertIs(importlib.import_module('image_search'),
                      importlib.import_module('foodquality8.dataset.image_search'))

    def test_unified_catalog_preserves_records_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'sample.jpg'
            image.write_bytes(b'sample')
            row = dict(path=str(image), food='tomato', condition='fresh', split='train',
                       source='Original dataset', sha256=hashlib.sha256(b'sample').hexdigest(),
                       duplicate_group='group-1')
            manifest = root / 'input.json'
            manifest.write_text(json.dumps([row]), encoding='utf8')
            result = consolidate([manifest, manifest], root / 'unified')
            self.assertEqual(result['images'], 1)
            self.assertEqual(json.loads((root/'unified/split.json').read_text(encoding='utf8')), [row])
            second = root / 'second.json'
            second.write_text(json.dumps([dict(row, split='test')]), encoding='utf8')
            with self.assertRaises(ValueError):
                consolidate([manifest, second], root / 'conflicting')
            self.assertFalse((root / 'conflicting').exists())
            image.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                consolidate([manifest], root / 'corrupted')


if __name__ == '__main__':
    unittest.main()

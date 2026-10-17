import unittest
from foodquality8.source_dataset import validate_rows
from foodquality8.food9 import FOODS

class SourceDatasetTests(unittest.TestCase):
    def rows(self):
        return [dict(food=f,condition=c,split=s,sha256=f+c+s,duplicate_group=f+c+s)
            for f in FOODS for c in ('fresh','spoiled') for s in ('train','val','test')]
    def test_leaked_group_is_rejected_even_with_different_files(self):
        rows=self.rows();rows[1]['duplicate_group']=rows[0]['duplicate_group']
        with self.assertRaisesRegex(ValueError,'trùng'):validate_rows(rows)
    def test_one_sided_freshness_cannot_be_trained(self):
        with self.assertRaisesRegex(ValueError,'Thiếu lớp'):validate_rows(self.rows()[:-1])
    def test_complete_disjoint_release_passes(self):
        self.assertEqual(len(validate_rows(self.rows())),54)

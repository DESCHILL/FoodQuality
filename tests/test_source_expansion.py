import unittest
from scripts.expand_source_dataset import parent_name,capture_date,assign_date,HashIndex


class SourceExpansionTests(unittest.TestCase):
    def test_source_variants_share_parent_and_capture_date(self):
        self.assertEqual(parent_name('aug_12_IMG-20260128-WA0015.jpg'),parent_name('IMG-20260128-WA0015.jpg'))
        self.assertEqual(capture_date('aug_12_IMG-20260128-WA0015.jpg'),'20260128')
        self.assertEqual(capture_date('IMG_20251104_131910090_HDR.jpg'),'20251104')
        self.assertIsNone(capture_date('IMG_20251345_bad.jpg'))
        self.assertIsNone(capture_date('Banana (9).jpeg'))

    def test_existing_capture_day_keeps_its_holdout(self):
        old={'20260128':('test','original_day_group')}
        self.assertEqual(assign_date('20260128',old),('test','original_day_group'))
        self.assertEqual(assign_date('20260129',old),assign_date('20260129',{}))

    def test_near_duplicate_index_spans_bit_boundaries(self):
        index=HashIndex();value=0x123456789abcdef0;index.add(value,'old-image')
        self.assertEqual(index.matches(value^(1<<1)^(1<<16)^(1<<30)^(1<<50)),['old-image'])
        self.assertEqual(index.matches(value^0xffff),[])

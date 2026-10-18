import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from foodquality8.food9 import FOODS
from foodquality8.robust_views import DETAIL, LEGACY, image_views, extract_features, feature_signature
from foodquality8.source_dataset import validate_rows
from scripts.build_robust_dataset import build


class RobustDatasetTests(unittest.TestCase):
    def test_promotion_rejects_food_or_spoilage_regressions(self):
        from scripts.compare_robust_model import passes
        base=dict(macro_f1=.9,food_accuracy=.98,spoiled_recall=.95)
        self.assertFalse(passes(base,dict(base,macro_f1=.92,spoiled_recall=.94)))
        self.assertFalse(passes(base,dict(base,macro_f1=.92,food_accuracy=.97)))
        self.assertTrue(passes(base,dict(base,macro_f1=.92)))

    def test_two_views_preserve_edges_and_double_pixel_area(self):
        im=Image.new('RGB',(600,200),'red')
        im.paste('blue',(0,0,25,200))
        views=image_views(im,DETAIL)
        self.assertEqual(len(views),2)
        self.assertEqual([v.size for v in views],[(320,320)]*2)
        self.assertTrue(np.any(np.all(np.asarray(views[0])==[0,0,255],axis=2)))
        self.assertFalse(np.array_equal(views[0],views[1]))
        self.assertAlmostEqual(320**2/224**2,2,delta=.05)

    def test_pooling_and_cache_recipe(self):
        class Encoder(torch.nn.Module):
            def forward(self,x):return x.mean((2,3))
        images=[Image.new('RGB',(400,200),'red'),Image.new('RGB',(200,400),'green')]
        features=extract_features(Encoder(),images,DETAIL)
        self.assertEqual(tuple(features.shape),(2,54))
        self.assertTrue(torch.isfinite(features).all())
        fast=extract_features(Encoder(),images,DETAIL,channels_last=True)
        self.assertTrue(torch.allclose(features,fast,atol=1e-5,rtol=1e-5))
        rows=[dict(sha256='a',split='train')]
        self.assertNotEqual(feature_signature(rows,DETAIL),feature_signature(rows,LEGACY))

    def test_derived_images_never_enter_holdouts_and_keep_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            rng=np.random.default_rng(42)
            for split in ('train','val','test'):
                for food in FOODS:
                    for condition in ('fresh','spoiled'):
                        path=root/f'{split}_{food}_{condition}.png'
                        Image.fromarray(rng.integers(0,256,(40,60,3),dtype=np.uint8)).save(path)
                        sha=hashlib.sha256(path.read_bytes()).hexdigest()
                        rows.append(dict(path=str(path),sha256=sha,duplicate_group=sha,
                                         food=food,condition=condition,split=split))
            report=build(rows,root/'derived')
            import json
            result=json.loads((root/'derived/split.json').read_text())
            self.assertEqual(report['derived_images'],18)
            self.assertEqual([r for r in result if r['split']!='train'],[r for r in rows if r['split']!='train'])
            self.assertEqual(report,build(rows,root/'derived'))
            from unittest.mock import patch
            with patch('scripts.build_robust_dataset.augment',side_effect=AssertionError('Reuse existing derivatives')):
                self.assertEqual(report,build(rows,root/'reused',[root/'derived']))
            result[-1]['split']='val'
            with self.assertRaises(ValueError):validate_rows(result)


if __name__=='__main__':unittest.main()

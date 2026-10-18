import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from foodquality8.feature_cache import ensure_features,atomic_cache
from foodquality8.robust_views import LEGACY,DETAIL,feature_signature


class FeatureCacheTests(unittest.TestCase):
    def test_only_missing_images_are_extracted_and_wrong_recipe_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'old';source.mkdir();dest=root/'new'
            path=root/'a.jpg';Image.new('RGB',(100,100),'green').save(path)
            rows=[dict(sha256=str(i),split='train',path=str(path)) for i in range(3)]
            (source/'split.json').write_text(json.dumps(rows[:2]))
            atomic_cache(source/'features.npz',features=np.ones((2,1331),dtype=np.float32),signature=feature_signature(rows[:2],DETAIL))
            with patch('foodquality8.feature_cache.extract_features',return_value=torch.full((1,1331),2.)) as extract:
                values=ensure_features(dest,rows,None,DETAIL,[source],log=lambda _:None)
                self.assertEqual(extract.call_count,1)
                self.assertTrue(np.all(values[:2]==1));self.assertTrue(np.all(values[2]==2))
            with patch('foodquality8.feature_cache.extract_features',side_effect=AssertionError('must reuse')):
                self.assertTrue(np.array_equal(values,ensure_features(dest,rows,None,DETAIL)))
            with self.assertRaises(ValueError):ensure_features(root/'wrong',rows,None,LEGACY,[source])

    def test_resume_only_uses_matching_partial_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'a.jpg';Image.new('RGB',(100,100)).save(image)
            rows=[dict(sha256=str(i),split='train',path=str(image)) for i in range(2)]
            atomic_cache(root/'features.partial.npz',features=np.ones((2,1331),dtype=np.float32),ready=np.array([True,False]),signature=feature_signature(rows,LEGACY))
            with patch('foodquality8.feature_cache.extract_features',return_value=torch.zeros((1,1331))) as extract:
                result=ensure_features(root,rows,None,LEGACY,log=lambda _:None)
                self.assertEqual(extract.call_count,1);self.assertTrue(np.all(result[0]==1))
                self.assertTrue(np.all(result[1]==0));self.assertFalse((root/'features.partial.npz').exists())

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
import torch
from PIL import Image
from foodquality8.consensus import ConsensusPredictor
from foodquality8.evaluation_cache import align_features
from foodquality8.feature_cache import atomic_cache
from foodquality8.robust_views import LEGACY,DETAIL,feature_signature


class EvaluationCacheTests(unittest.TestCase):
    def test_alignment_uses_hashes_and_rejects_wrong_recipe_or_missing_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[dict(sha256=str(i),split='val') for i in range(3)]
            (root/'split.json').write_text(json.dumps(rows))
            features=np.stack([np.full(1331,i,dtype=np.float32) for i in range(3)])
            atomic_cache(root/'features.npz',features=features,signature=feature_signature(rows,DETAIL))
            result=align_features(root,DETAIL,[rows[2],rows[0]])
            np.testing.assert_array_equal(result,features[[2,0]])
            with self.assertRaises(ValueError):align_features(root,LEGACY,rows)
            with self.assertRaises(ValueError):align_features(root,DETAIL,[dict(sha256='missing')])

    def test_cached_inference_keeps_detector_and_quality_decision_without_encoder(self):
        model=object.__new__(ConsensusPredictor)
        classifier=torch.nn.Linear(1331,9)
        with torch.no_grad():
            classifier.weight.zero_();classifier.bias.zero_();classifier.bias[0]=5
        model.base=SimpleNamespace(preprocessing=DETAIL,food=classifier,
                                   data={'mean':torch.zeros(1331),'std':torch.ones(1331)},
                                   encoder=Mock(side_effect=AssertionError('Encoder must be reused')))
        model.sic={'model_state_dict':{'classifier.1.weight':torch.zeros(2,1280),
                                       'classifier.1.bias':torch.tensor([3.,-3.])}}
        model.names=['apple__normal_appearance','apple__defect']
        model.detector=SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=[],names={})]))
        model.config={'calibration':{'foodquality_temperature':1.,'foodquality_quality_temperature':1.,
                                     'sic_temperature':1.,'sic_quality_temperature':1.,'yolo_platt':[1.,0.]}}
        logits=torch.zeros(1,9,2);logits[:,:,0]=5
        with patch('foodquality8.quality_ensemble.condition_logits',return_value=logits):
            result=model.predict_cached(Image.new('RGB',(40,40)),np.zeros(1331),np.zeros(1331))[0]
        self.assertEqual((result['food'],result['condition']),('apple','fresh'))
        model.base.encoder.assert_not_called();model.detector.predict.assert_called_once()
        with self.assertRaises(ValueError):model.predict_cached(None,np.zeros(2),np.zeros(1331))
        with self.assertRaises(ValueError):model.predict_cached(None,np.full(1331,np.nan),np.zeros(1331))


if __name__=='__main__':unittest.main()

import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from PIL import Image
from foodquality8.fusion import FusionPredictor,canonical_food,reconcile

class FusionTests(unittest.TestCase):
    def test_cooked_names_do_not_become_raw_foods(self):
        self.assertIsNone(canonical_food('beef-curry'))
        self.assertIsNone(canonical_food('Grilled Chicken'))
        self.assertEqual(canonical_food('Apple'),'apple')

    def test_conflicting_models_abstain(self):
        row=dict(food='carrot',condition='fresh',quality_score=.99,quality_candidates=[1],tentative=False)
        result=reconcile(row,'orange',.95,[0,0,100,100],(100,100))
        self.assertEqual(result['food'],'orange')
        self.assertEqual(result['condition'],'unknown')
        self.assertEqual(result['quality_candidates'],[])
        self.assertEqual(row['condition'],'fresh')

    def test_tiny_crop_cannot_get_a_freshness_claim(self):
        row=dict(food='apple',condition='fresh')
        self.assertEqual(reconcile(row,'apple',.9,[0,0,10,10],(10,10))['condition'],'unknown')

    def test_confident_primary_does_not_get_overridden_by_detector(self):
        model=object.__new__(FusionPredictor)
        model.config={'routing':'classification_first'}
        model.quality=Mock()
        model.quality.predict.return_value=[dict(food='apple',condition='fresh',tentative=False)]
        model.detector=Mock()
        result=model.predict(Image.new('RGB',(64,64)))
        self.assertEqual(result[0]['food'],'apple')
        model.detector.predict.assert_not_called()

    def test_uncertain_primary_uses_detector_but_no_boxes_stay_unknown(self):
        model=object.__new__(FusionPredictor)
        model.config={'routing':'classification_first'}
        model.quality=Mock()
        model.quality.predict.return_value=[dict(food='apple',condition='unknown',tentative=True,reason='Điểm thấp')]
        model.detector=Mock()
        model.detector.predict.return_value=[SimpleNamespace(boxes=[])]
        result=model.predict(Image.new('RGB',(64,64)))
        self.assertEqual(result[0]['condition'],'unknown')
        model.detector.predict.assert_called_once()

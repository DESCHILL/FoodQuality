import unittest
import numpy as np
from PIL import Image
from foodquality8.food9 import FOODS,visual_features


class Food9Tests(unittest.TestCase):
    def test_scope_is_nine_foods_without_pork(self):
        self.assertEqual(len(set(FOODS)),9)
        self.assertEqual(set(FOODS)&{'beef','chicken','mutton','pork'},{'beef','chicken','mutton'})

    def test_visual_features_describe_color_not_a_spoilage_rule(self):
        white=visual_features(Image.new('RGB',(128,128),'white'))
        black=visual_features(Image.new('RGB',(128,128),'black'))
        self.assertEqual(len(white),51)
        self.assertEqual(float(white[-2]),1.)
        self.assertEqual(float(black[-3]),1.)
        self.assertTrue(np.isfinite(white).all())

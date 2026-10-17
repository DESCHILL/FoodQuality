import unittest
from types import SimpleNamespace
from foodquality8.inference import select_detections


def box(score):
    return SimpleNamespace(conf=SimpleNamespace(item=lambda:score))


class TentativeDetectionTests(unittest.TestCase):
    def test_borderline_box_is_visible_but_not_confirmed(self):
        candidate=box(.3364695)
        boxes,tentative=select_detections([candidate])
        self.assertEqual(boxes,[candidate])
        self.assertTrue(tentative)

    def test_confident_boxes_take_precedence(self):
        low,high=box(.2),box(.8)
        boxes,tentative=select_detections([low,high])
        self.assertEqual(boxes,[high])
        self.assertFalse(tentative)

    def test_no_box_is_invented(self):
        self.assertEqual(select_detections([]),([],False))
        self.assertEqual(select_detections([box(.05)]),([],False))

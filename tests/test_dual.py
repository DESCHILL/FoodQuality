import unittest
from foodquality8.ai_detection.dual import combine


class DualPipelineTests(unittest.TestCase):
    def test_quality_is_not_borrowed_from_another_food(self):
        sources=[dict(food='apple',score=.95),dict(food='beef',score=.8)]
        evidence=[dict(food='beef',food_score=.9,condition='fresh',score=.99)]
        result=combine(sources,evidence,.8)
        self.assertEqual(result['food'],'apple')
        self.assertEqual(result['condition'],'unknown')

    def test_low_confidence_does_not_force_a_quality_label(self):
        result=combine([dict(food='tomato',score=.5)],
                       [dict(food='tomato',food_score=.5,condition='fresh',score=.99)],.8)
        self.assertEqual(result['condition'],'unknown')

    def test_unsupported_detection_does_not_get_freshness(self):
        result=combine([dict(food='pizza',score=.99)],
                       [dict(food='pizza',food_score=.99,condition='fresh',score=.99)],.8)
        self.assertEqual(result['condition'],'unknown')

    def test_calibrated_matching_evidence_is_used(self):
        result=combine([dict(food='tomato',score=.9)],
                       [dict(food='tomato',food_score=.9,condition='spoiled',score=.91)],.8)
        self.assertEqual(result['condition'],'spoiled')


if __name__=='__main__':unittest.main()

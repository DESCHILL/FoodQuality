import unittest
from foodquality8.consensus import select_recognition,decide_quality

class ConsensusTests(unittest.TestCase):
    def test_highest_score_is_selected_after_collecting_all_sources(self):
        sources=[dict(source='FoodQuality',food='apple',score=.7),dict(source='SIC EfficientNet',food='tomato',score=.92),dict(source='SIC YOLO',food='tomato',score=.6)]
        self.assertEqual(select_recognition(sources)['food'],'tomato')
        self.assertEqual(select_recognition(list(reversed(sources)))['food'],'tomato')

    def test_no_detection_is_not_a_vote_for_a_food(self):
        self.assertEqual(select_recognition([dict(food=None,score=1),dict(food='tomato',score=.5)])['food'],'tomato')

    def test_disagreement_cannot_become_a_spoilage_claim(self):
        evidence=[dict(food='tomato',food_score=.9,score=.9,condition='fresh'),dict(food='tomato',food_score=.9,score=.95,condition='spoiled')]
        self.assertEqual(decide_quality('tomato',evidence)[0],'unknown')

    def test_quality_of_another_food_is_not_reused(self):
        self.assertEqual(decide_quality('tomato',[dict(food='apple',food_score=.99,score=.99,condition='fresh')])[0],'unknown')

    def test_generic_defect_does_not_conflict_with_separate_spoilage_evidence(self):
        evidence=[dict(food='tomato',food_score=.9,score=.9,condition='spoiled'),dict(food='tomato',food_score=.9,score=.95,condition='defect')]
        self.assertEqual(decide_quality('tomato',evidence)[0],'spoiled')
        self.assertEqual(decide_quality('tomato',evidence[1:])[0],'defect')

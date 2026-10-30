import tempfile
import unittest
from pathlib import Path

from foodquality8.model_data import write_json
from foodquality8.model_selection import validation_improved


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.old,self.new=Path(self.temp.name)/'old',Path(self.temp.name)/'new'
        for p in (self.old,self.new):
            write_json(p/'run.json',dict(dataset=dict(dataset_sha256='same')))
            write_json(p/'validation.json',dict(macro_f1=.6,loss=.7,per_class={'pork_spoiled':{'recall':.7}}))

    def test_loss_alone_is_not_enough_and_test_metrics_do_not_select(self):
        write_json(self.new/'validation.json',dict(macro_f1=.6,loss=.3))
        write_json(self.new/'test-metrics.json',dict(accuracy=1.,macro_f1=1.))
        self.assertFalse(validation_improved(self.new,self.old,'classifier')[0])

    def test_better_macro_score_cannot_hide_worse_spoiled_recall(self):
        write_json(self.new/'validation.json',dict(macro_f1=.9,loss=.3,per_class={'pork_spoiled':{'recall':.5}}))
        self.assertFalse(validation_improved(self.new,self.old,'classifier')[0])
        write_json(self.new/'validation.json',dict(macro_f1=.8,loss=.3,per_class={'pork_spoiled':{'recall':.7}}))
        self.assertTrue(validation_improved(self.new,self.old,'classifier')[0])

    def test_different_data_cannot_be_compared_as_same_holdout(self):
        write_json(self.new/'run.json',dict(dataset=dict(dataset_sha256='different')))
        self.assertFalse(validation_improved(self.new,self.old,'classifier')[0])

    def test_training_new_food_switches_whole_pair_not_old_food(self):
        from foodquality8.model_selection import choose_bundle
        old={'classifier':{'class_labels':[{'food':'pork'}]},'detector':{'classes':['pork']}}
        new={'classifier':{'class_labels':[{'food':'apple'}]},'detector':{'classes':['apple']}}
        chosen,reason=choose_bundle(new,old)
        self.assertEqual(chosen,new)
        self.assertIn('pair',reason)

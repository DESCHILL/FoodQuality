import unittest
import torch
from foodquality8.quality_ensemble import condition_logits


class QualityEnsembleTests(unittest.TestCase):
    def head(self,bias):
        return dict(mean=torch.zeros(3),std=torch.ones(3),
                    quality_head=dict(weight=torch.zeros(18,3),bias=torch.tensor(bias*9,dtype=torch.float32)))

    def test_quality_probabilities_mix_without_changing_food_identity(self):
        new=self.head([0.,4.]);old=self.head([3.,0.])
        new['legacy_quality']=dict(old,weight=.5)
        result=condition_logits(new,torch.zeros(2,3),torch.zeros(2,3)).softmax(-1)
        expected=(torch.tensor([0.,4.]).softmax(0)+torch.tensor([3.,0.]).softmax(0))*.5
        self.assertTrue(torch.allclose(result,expected.expand(2,9,2)))
        self.assertLess(float(result.max()),.6)

    def test_legacy_input_cannot_be_silently_replaced_with_detail_features(self):
        ck=self.head([0.,4.]);ck['legacy_quality']=dict(self.head([3.,0.]),weight=.5)
        with self.assertRaises(ValueError):condition_logits(ck,torch.zeros(1,3))

    def test_existing_checkpoint_logits_remain_compatible(self):
        ck=self.head([1.,2.]);x=torch.tensor([[1.,4.,-2.]])
        expected=torch.nn.functional.linear(x,ck['quality_head']['weight'],ck['quality_head']['bias']).reshape(1,9,2)
        self.assertTrue(torch.equal(condition_logits(ck,x),expected))

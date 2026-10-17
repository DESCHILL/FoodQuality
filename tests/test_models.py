import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from foodquality8.model_data import check_release, write_json
from foodquality8.inference import activate_run, load_bundle, quality_decision


def fixture_release(root):
    rows=[]
    for s,split in enumerate(('train','val','test')):
        for i,label in enumerate(('apple_fresh','apple_spoiled')):
            iid=f'{split}_{i}'
            path=root/'classification'/split/label/(iid+'.jpg')
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.random.default_rng(s*10+i).integers(0,255,(240,240,3),dtype=np.uint8)).save(path)
            row=dict(image_id=iid,food_name='apple',condition=label.split('_')[1],review_status='Approved',
                     source_dataset='fixture',split=split,group_id=iid,capture_group=iid,pixel_hash=iid,sha256=iid,
                     bbox_status='Approved',bbox_json=json.dumps([dict(food_name='apple',x=.5,y=.5,w=.5,h=.5)]))
            rows.append(row)
            dest=root/'detection/images'/split/path.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(path.read_bytes())
            ann=root/'detection/labels'/split/(iid+'.txt')
            ann.parent.mkdir(parents=True,exist_ok=True)
            ann.write_text('0 0.5 0.5 0.5 0.5',encoding='utf8')
    (root/'detection/data.yaml').write_text('names: [apple]\n',encoding='utf8')
    with (root/'metadata.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root/'manifest.json',dict(classes=['apple_fresh','apple_spoiled'],
        catalog_sha256=hashlib.sha256((root/'metadata.csv').read_bytes()).hexdigest()))
    write_json(root/'sources.json',{'fixture':{'license_status':'Approved'}})
    return root


class ModelTests(unittest.TestCase):
    def test_inference_preprocessing_matches_training_validation(self):
        import torch
        from torchvision.models import EfficientNet_B0_Weights
        from foodquality8.inference import quality_transform
        transform=EfficientNet_B0_Weights.DEFAULT.transforms()
        prep=dict(resize=256,crop=224,mean=transform.mean,std=transform.std,interpolation=transform.interpolation.value)
        image=Image.fromarray(np.random.default_rng(100).integers(0,255,(300,480,3),dtype=np.uint8))
        self.assertTrue(torch.equal(transform(image),quality_transform(prep)(image)))

    def test_release_validation_and_pixel_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture_release(Path(tmp))
            self.assertEqual(check_release(root,'classifier')['classes'],['apple_fresh','apple_spoiled'])
            self.assertEqual(check_release(root,'detector')['classes'],['apple'])
            train=root/'classification/train/apple_fresh/train_0.jpg'
            val=root/'classification/val/apple_fresh/val_0.jpg'
            val.write_bytes(train.read_bytes())
            with self.assertRaisesRegex(ValueError,'pixel'):
                check_release(root,'classifier')

    def test_yolo_boxes_and_unapproved_source_block_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture_release(Path(tmp))
            (root/'detection/labels/train/train_0.txt').write_text('0 0.99 0.5 0.5 0.5')
            with self.assertRaisesRegex(ValueError,'Bounding box'):
                check_release(root,'detector')
            write_json(root/'sources.json',{'fixture':{'license_status':'Pending'}})
            with self.assertRaisesRegex(ValueError,'quyen'):
                check_release(root,'classifier')

    def test_conflicting_models_and_low_scores_reject(self):
        labels=[dict(food='beef',condition='fresh'),dict(food='apple',condition='spoiled')]
        self.assertEqual(quality_decision([.9,.1],labels,'apple',.6)['condition'],'unknown')
        self.assertEqual(quality_decision([.5,.5],labels,'beef',.6)['condition'],'unknown')
        self.assertEqual(quality_decision([.1,.9],labels,'apple',.6)['condition'],'spoiled')

    def test_incomplete_run_cannot_activate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'run.json',{'status':'stopped'})
            with self.assertRaises(ValueError):
                activate_run(root,root/'active.json')
            with self.assertRaises(ValueError):
                load_bundle(root/'active.json')

    def test_classifier_cpu_smoke_saves_reloadable_checkpoint_and_metrics(self):
        from foodquality8.training import train_classifier
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture_release(Path(tmp)/'release')
            info=check_release(root,'classifier')
            run=Path(tmp)/'run'
            run.mkdir()
            self.assertTrue(train_classifier(info,run,1,2,'cpu',run/'STOP',pretrained=False))
            checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            self.assertEqual(checkpoint['classes'],info['classes'])
            self.assertEqual(json.loads((run/'test-metrics.json').read_text())['samples'],2)
            (run/'STOP').touch()
            self.assertFalse(train_classifier(info,run,1,2,'cpu',run/'STOP',pretrained=False))


if __name__=='__main__':
    unittest.main()

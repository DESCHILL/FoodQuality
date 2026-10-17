import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from foodquality8 import Dataset
from foodquality8.annotations import box_from_drag,save_boxes,save_source_review
from foodquality8.google_review import save_review


class AnnotationTests(unittest.TestCase):
    def test_drag_normalization_reverse_outside_and_tiny(self):
        box=box_from_drag((110,110),(-10,-10),100,100,'apple')
        self.assertEqual((box['x'],box['y'],box['w'],box['h']),(.5,.5,1,1))
        with self.assertRaises(ValueError):
            box_from_drag((1,1),(2,2),100,100,'apple')

    def test_box_and_source_approval_require_evidence_and_reset_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/'raw/FreshApple/a.png'
            image.parent.mkdir(parents=True)
            Image.new('RGB',(200,100),'green').save(image)
            source=dict(id='own',source_url='https://example.org/a',license_status='Pending',mapping={'FreshApple':['apple','fresh']})
            cfg=root/'source.json'
            cfg.write_text(json.dumps({'sources':[source]}))
            d=Dataset(root/'dataset')
            try:
                d.configure(cfg)
                d.ingest('own',root/'raw')
                row=d.rows()[0]
                box=box_from_drag((0,0),(100,50),200,100,'apple')
                with self.assertRaises(ValueError):
                    save_boxes(d,row['image_id'],[box],'')
                save_boxes(d,row['image_id'],[box],'tester')
                self.assertEqual(d.rows()[0]['bbox_status'],'Approved')
                with self.assertRaises(ValueError):
                    save_source_review(d,'own','Approved')
                save_source_review(d,'own','Approved','fixture','https://example.org/license','https://example.org/a','tester','TEST ONLY')
                self.assertEqual(d.sources()['own']['license_status'],'Approved')
                self.assertFalse(d.sources()['own']['redistribution_allowed'])
                save_review(d,row['image_id'],'banana','fresh','Pending')
                self.assertEqual(d.rows()[0]['bbox_status'],'Pending')
                save_source_review(d,'own','Rejected')
                self.assertEqual(d.sources()['own']['license_status'],'Rejected')
            finally:
                d.close()


if __name__=='__main__':
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage

from foodquality8.auto_workflow import import_local,stage_name,propose_box,add_auto_boxes
from foodquality8.quick_dataset import prepare_google
from foodquality8.model_data import check_release,write_json
from foodquality8.train_pair import run_pair


class AutoWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'input'
        for j,stage in enumerate(('Tuoi','Hong')):
            folder=self.source/stage
            folder.mkdir(parents=True)
            for i in range(3):
                Image.fromarray(np.random.default_rng(j*10+i).integers(0,256,(300,300,3),dtype=np.uint8)).save(folder/f'{i}.jpg')

    def test_folder_and_automatic_release_for_both_models(self):
        folder=import_local(self.root,self.source,'thịt bò')
        original=(self.source/'Tuoi/0.jpg').read_bytes()
        release=prepare_google(self.root,'thịt bò',source_folder=folder,required_stages=['fresh','spoiled'])
        with patch('foodquality8.auto_workflow.propose_box',return_value=dict(x=.5,y=.5,w=.8,h=.8)):
            combined=add_auto_boxes(release,log=lambda s:None)
        for kind in ('classifier','detector'):
            info=check_release(combined,kind,experimental=True)
            self.assertEqual(info['label_mode'],'automatic_suggested')
            with self.assertRaises(ValueError):
                check_release(combined,kind)
        self.assertEqual((self.source/'Tuoi/0.jpg').read_bytes(),original)
        self.assertFalse((Path(release)/'detection').exists())

    def test_real_foreground_proposal_and_blank_rejection(self):
        arr=np.full((300,300,3),245,np.uint8)
        arr[70:230,85:215]=np.random.default_rng(12).integers(30,160,(160,130,3),dtype=np.uint8)
        box=propose_box(Image.fromarray(arr))
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box['x'],.5,delta=.1)
        self.assertLess(box['w'],.7)
        self.assertIsNone(propose_box(Image.new('RGB',(300,300),'white')))
        self.assertIsNone(stage_name('semi_fresh'))
        self.assertIsNone(stage_name('rat_hong'))

    def test_excel_embedded_images_without_original_paths(self):
        wb=Workbook()
        ws=wb.active
        ws.append(['food','quality','image'])
        ws.append(['beef','fresh',''])
        ws.append(['apple','spoiled',''])
        ws.add_image(ExcelImage(str(self.source/'Tuoi/0.jpg')),'C2')
        ws.add_image(ExcelImage(str(self.source/'Hong/0.jpg')),'C3')
        path=self.root/'input.xlsx'
        wb.save(path)
        folder=import_local(self.root,path,'thịt bò')
        rows=[json.loads(s) for s in (folder/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['search_stage'],'fresh')

    def test_pair_order_and_atomic_activation(self):
        run=self.root/'run'
        order=[]
        def launch(args,**kwargs):
            kind=args[4]
            order.append(kind)
            child=run/kind
            write_json(child/'run.json',{'status':'complete'})
            return Mock(poll=Mock(return_value=0),returncode=0)
        def activate(child,target,**kwargs):
            state=json.loads(target.read_text()) if target.exists() else {}
            state[child.name]={'experimental':True}
            write_json(target,state)
        with patch('foodquality8.quick_dataset.prepare_google',return_value=self.root/'release'), \
             patch('foodquality8.auto_workflow.add_auto_boxes',return_value=self.root/'combined'), \
             patch('foodquality8.train_pair.subprocess.Popen',side_effect=launch), \
             patch('foodquality8.inference.activate_run',side_effect=activate):
            result=run_pair(self.root,'google','beef',run,epochs=1)
        self.assertEqual(order,['classifier','detector'])
        self.assertEqual(result['status'],'complete')
        self.assertEqual(set(json.loads((self.root/'models/active.json').read_text())),{'classifier','detector'})

    def test_failure_does_not_run_second_model_or_change_active(self):
        active=self.root/'models/active.json'
        write_json(active,{'previous':True})
        with patch('foodquality8.quick_dataset.prepare_google',return_value=self.root/'release'), \
             patch('foodquality8.auto_workflow.add_auto_boxes',return_value=self.root/'combined'), \
             patch('foodquality8.train_pair.subprocess.Popen',return_value=Mock(poll=Mock(return_value=1),returncode=1)) as launch:
            with self.assertRaises(ValueError):
                run_pair(self.root,'google','beef',self.root/'run')
        self.assertEqual(launch.call_count,1)
        self.assertEqual(json.loads(active.read_text()),{'previous':True})

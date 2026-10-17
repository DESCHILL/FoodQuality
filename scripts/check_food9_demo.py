"""Smoke-check real checkpoint and automatic upload flow, without network calls."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from streamlit.testing.v1 import AppTest
from foodquality8.food9 import Food9Predictor,FOODS

rows=json.loads((ROOT/'food9_workspace/split.json').read_text(encoding='utf8'))
groups={s:{r['duplicate_group'] for r in rows if r['split']==s} for s in ('train','val','test')}
assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
predictor=Food9Predictor(ROOT/'models/food9.json')
results=[]
for food in FOODS:
    row=next(r for r in rows if r['split']=='test' and r['food']==food)
    with Image.open(row['path']) as image:
        prediction=predictor.predict(image)[0]
    results.append(dict(expected=food,predicted=prediction['food'],quality=prediction['condition']))
    assert prediction['food'] in FOODS
    assert prediction['condition'] in ('fresh','spoiled','unknown')
apple=next(r for r in rows if r['split']=='test' and r['food']=='apple')
class Upload(io.BytesIO):
    name='apple.jpg'
    @property
    def size(self): return len(self.getvalue())
def upload(*args,**kwargs): return Upload(Path(apple['path']).read_bytes())
with patch('streamlit.file_uploader',side_effect=upload):
    app=AppTest.from_file(str(ROOT/'user_app.py')).run(timeout=60)
    assert not app.exception, str(app.exception)
    assert not app.error, str(app.error)
    assert app.session_state['analysis_rows'][0]['food']=='apple'
    app.run(timeout=60)
    assert len(app.session_state['history'])==1
report=dict(samples=results,automatic_upload='passed',apple_recognition='passed',history_cache='passed',split_groups='disjoint')
(ROOT/'model_runs/food9/source-binary-v1/smoke-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False,indent=2))

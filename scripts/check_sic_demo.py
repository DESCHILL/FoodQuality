"""Real-source image smoke check and automatic web-upload integration check."""
import argparse,io,json,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from streamlit.testing.v1 import AppTest
from foodquality8.fusion import FusionPredictor

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);args=p.parse_args()
    model=FusionPredictor(ROOT/'models/fusion.json');results=[]
    for name in ('apple_01.jpg','banana_01.jpg','orange_01.jpg'):
        path=args.repo/'images/real_test'/name
        with Image.open(path) as im:pred=model.predict(im)
        results.append(dict(image=name,predictions=pred))
    class Upload(io.BytesIO):
        name='apple.jpg'
        @property
        def size(self):return len(self.getvalue())
    raw=(args.repo/'images/real_test/apple_01.jpg').read_bytes()
    with patch('streamlit.file_uploader',side_effect=lambda *a,**k:Upload(raw)):
        app=AppTest.from_file(str(ROOT/'user_app.py')).run(timeout=60)
        assert not app.exception and not app.error
        assert app.selectbox[0].value=='Kết hợp SIC + FoodQuality'
        assert app.session_state['analysis_rows'][0]['food']=='apple'
        app.run(timeout=60)
        assert len(app.session_state['history'])==1
    report=dict(ui='passed',samples=results,note='Smoke checks only, not an accuracy benchmark.')
    (ROOT/'workspace/sic_merged/web-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

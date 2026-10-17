"""Verify automatic upload, all three sources and the reported regression image."""
import argparse,io,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',type=Path,required=True);args=parser.parse_args()
    class Upload(io.BytesIO):
        name='regression.jpg'
        @property
        def size(self):return len(self.getvalue())
    raw=args.image.read_bytes()
    with patch('streamlit.file_uploader',side_effect=lambda *a,**k:Upload(raw)):
        app=AppTest.from_file(str(ROOT/'user_app.py')).run(timeout=60)
        assert not app.exception and not app.error
        assert app.selectbox[0].value=='Nhận diện từ cả 3 nguồn'
        result=app.session_state['analysis_rows'][0]
        assert len(result['source_scores'])==3
        assert result['food']=='tomato'
        assert result['condition']!='spoiled'
        app.run(timeout=60)
        assert len(app.session_state['history'])==1
        print('Automatic upload, three-source recognition and tomato regression: PASS')

if __name__=='__main__':main()

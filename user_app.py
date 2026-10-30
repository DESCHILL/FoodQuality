"""End-user interface. Contains no dataset collection or training controls."""
from foodquality8 import module_path
import hashlib
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from foodquality8.inference import FoodPredictor, load_bundle

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'models/active.json'
FOODS = dict(beef='Thịt bò', pork='Thịt lợn', chicken='Thịt gà', mutton='Thịt cừu', apple='Táo', banana='Chuối',
             orange='Cam', mango='Xoài', strawberry='Dâu tây', grape='Nho', guava='Ổi',
             tomato='Cà chua', potato='Khoai tây', cucumber='Dưa chuột', carrot='Cà rốt', bell_pepper='Ớt chuông')
CONDITIONS = dict(fresh='Tươi', fresh_raw='Tươi sống', semi_fresh='Ít tươi', less_fresh='Ít tươi',
                  spoiled='Có dấu hiệu hỏng', visibly_spoiled='Có dấu hiệu hỏng', unknown='Chưa xác định',
                  cooked='Đã nấu', near_spoilage='Sắp hỏng', near_spoiled='Sắp hỏng', severely_spoiled='Hỏng nặng')

st.set_page_config(page_title='FoodQuality', page_icon='🥬', layout='wide')
st.markdown('''<style>
.block-container{max-width:1150px;padding-top:2.5rem}
[data-testid="stAppViewContainer"]{background:#f7faf8}
[data-testid="stMain"]{color:#18392d}
[data-testid="stMain"] p,[data-testid="stMain"] label{color:#18392d}
h1,h2,h3{color:#164c3d}
[data-testid="stToolbar"]{display:none}
[data-testid="stMetric"]{background:white;border:1px solid #dce9e1;border-radius:12px;padding:18px}
.food-banner{position:relative;overflow:hidden;background:linear-gradient(115deg,#123d2d,#22704e);border-radius:24px;padding:42px 40px;margin-bottom:34px;color:white;min-height:180px;box-sizing:border-box}
.food-banner h1{color:white!important;font-size:44px;margin:0;padding:0;letter-spacing:-1.5px}
.food-banner p{color:#e0f2e7!important;margin:12px 0 0;font-size:17px}
.food-banner .illustration{position:absolute;right:40px;top:32px;font-size:72px;letter-spacing:8px;transform:rotate(-8deg)}
.food-banner .banner-copy{position:relative;z-index:1;max-width:65%}
.food-footer{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;border-top:1px solid #dce9e1;margin-top:72px;padding:24px 0;color:#60796d;font-size:14px}
.food-footer strong{color:#164c3d;font-size:20px;letter-spacing:-.5px}
.food-footer span{color:#60796d}
[data-testid="stMainBlockContainer"]{min-height:100vh;display:flex;flex-direction:column;padding-bottom:16px}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]{flex:1}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.food-footer){margin-top:auto}
[data-testid="stFileUploader"]{background:white;border:1px solid #dce9e1;border-radius:16px;padding:12px}
@media(max-width:700px){.food-banner{padding:28px 24px}.food-banner h1{font-size:34px}.food-banner .banner-copy{max-width:100%}.food-banner .illustration{display:none}}
</style>''', unsafe_allow_html=True)
st.markdown('''<section class="food-banner"><div class="banner-copy">
<h1>FoodQuality</h1><p>Nhận diện thực phẩm · Đánh giá tươi, hỏng qua hình ảnh</p>
</div><div class="illustration" aria-hidden="true">🥬 🍅 🍊</div></section>''',unsafe_allow_html=True)

ready = False
food9_mode=False
fusion_mode=False
consensus_mode=False
dual_mode=os.environ.get('FOODQUALITY_PROFILE')=='dual'
consensus_mode=not dual_mode and (ROOT/'models/consensus.json').exists()
fusion_mode=not dual_mode and not consensus_mode and (ROOT/'models/fusion.json').exists()
food9_mode=dual_mode or consensus_mode or fusion_mode or (ROOT/'models/food9.json').exists()
if food9_mode:
    BUNDLE=ROOT/'models/dual.json' if dual_mode else ROOT/'models/consensus.json' if consensus_mode else ROOT/'models/fusion.json' if fusion_mode else ROOT/'models/food9.json'
try:
    if food9_mode:
        import json
        registry=json.loads(BUNDLE.read_text(encoding='utf8'))
        weight_registry=json.loads(Path(registry['quality_registry']).read_text(encoding='utf8')) if dual_mode or fusion_mode or consensus_mode else registry
        if not Path(weight_registry['weights']).is_file():
            raise ValueError('Thiếu trọng số demo 9 loại.')
        bundle={'detector':{'classes':registry['foods']},'classifier':{}}
    else:
        bundle = load_bundle(BUNDLE)
    ready = True
except (ValueError, OSError, KeyError) as exc:
    st.info(str(exc))

@st.cache_resource
def get_predictor(bundle_fingerprint, implementation_fingerprint, registry_path, use_food9):
    import importlib
    if use_food9=='dual':
        from foodquality8.ai_detection import dual
        return importlib.reload(dual).DualPredictor(registry_path)
    if use_food9=='consensus':
        from foodquality8 import consensus
        return importlib.reload(consensus).ConsensusPredictor(registry_path)
    if use_food9=='fusion':
        from foodquality8 import food9,fusion
        importlib.reload(food9)
        return importlib.reload(fusion).FusionPredictor(registry_path)
    if use_food9:
        from foodquality8 import food9
        return importlib.reload(food9).Food9Predictor(registry_path)
    from foodquality8 import inference
    return importlib.reload(inference).FoodPredictor(registry_path)

with st.container(border=True):
    st.markdown('**Thực phẩm được hỗ trợ**')
    supported = set(bundle['detector']['classes']) if ready else set()
    groups = [('🥩 Thịt', ('beef', 'chicken', 'mutton')),
              ('🍊 Trái cây', ('apple', 'banana', 'orange')),
              ('🥕 Rau củ', ('tomato', 'cucumber', 'carrot'))]
    for column, (title, foods) in zip(st.columns(3), groups):
        with column:
            names = [FOODS[food] for food in foods if food in supported]
            st.markdown(f'**{title}**')
            st.write(' · '.join(names) if names else 'Chưa có mô hình hỗ trợ')
    st.caption('Đánh giá: Tươi / Có dấu hiệu hỏng. Nên dùng ảnh rõ, một loại thực phẩm mỗi lần.')

left,right=st.columns([1,1],gap='large')
with left:
    st.subheader('1. Chọn ảnh thực phẩm')
    mode=st.radio('Chọn cách thêm ảnh',['Tải ảnh','Chụp ảnh'],horizontal=True,label_visibility='collapsed')
    uploaded=st.file_uploader('Ảnh JPG, PNG hoặc WEBP',type=['jpg','jpeg','png','webp']) if mode=='Tải ảnh' else st.camera_input('Chụp rõ thực phẩm')
    image=None
    if uploaded is not None:
        try:
            if uploaded.size > 15*1024*1024:
                raise ValueError('Vui lòng chọn ảnh dưới 15 MB.')
            original=Image.open(uploaded)
            if original.width*original.height>25_000_000:
                raise ValueError('Vui lòng chọn ảnh dưới 25 megapixel.')
            image=ImageOps.exif_transpose(original).convert('RGB')
            st.image(image,width='stretch')
        except Exception as exc:
            st.error(f'Không đọc được ảnh: {exc}')
    analyze=ready and image is not None
with right:
    st.subheader('2. Kết quả phân tích')
    if analyze:
        try:
            with st.spinner('Đang nhận diện và đánh giá...'):
                fingerprint=hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
                implementation_bytes=module_path('food9' if food9_mode else 'inference').read_bytes()
                if fusion_mode: implementation_bytes+=module_path('fusion').read_bytes()
                if consensus_mode: implementation_bytes+=module_path('consensus').read_bytes()
                if dual_mode: implementation_bytes+=module_path('ai_detection.dual').read_bytes()+module_path('consensus').read_bytes()
                if food9_mode or fusion_mode or consensus_mode:
                    implementation_bytes+=module_path('food9').read_bytes()+module_path('robust_views').read_bytes()+module_path('quality_ensemble').read_bytes()
                implementation=hashlib.sha256(implementation_bytes).hexdigest()
                request_key=hashlib.sha256(uploaded.getvalue()).hexdigest()+fingerprint+implementation
                fresh_result=st.session_state.get('analysis_key')!=request_key
                if fresh_result:
                    predictor=get_predictor(fingerprint,implementation,str(BUNDLE),'dual' if dual_mode else 'consensus' if consensus_mode else 'fusion' if fusion_mode else food9_mode)
                    rows=predictor.predict(image)
                    st.session_state['analysis_rows']=rows
                    st.session_state['analysis_key']=request_key
                else:
                    rows=st.session_state['analysis_rows']
            if not rows:
                st.warning('Chưa phát hiện thực phẩm thuộc phạm vi hỗ trợ. Hãy thử ảnh rõ hơn hoặc góc chụp khác.')
            else:
                annotated=image.copy()
                draw=ImageDraw.Draw(annotated)
                for i,row in enumerate(rows,1):
                    if row['bbox'] is None:
                        continue
                    draw.rectangle(row['bbox'],outline='#c47b16' if row.get('tentative') else '#16845b',width=max(2,image.width//300))
                    draw.text((row['bbox'][0]+5,row['bbox'][1]+5),str(i),fill='white',stroke_width=2,stroke_fill='#164c3d')
                st.image(annotated,width='stretch')
                for i,row in enumerate(rows,1):
                    with st.container(border=True):
                        prefix='Có thể là ' if row.get('tentative') else ''
                        st.markdown(f"**{i}. {prefix}{FOODS.get(row['food'],row['food'])}**")
                        st.write(CONDITIONS.get(row['condition'],row['condition']))
                        if row['condition']=='unknown':
                            st.caption(f"Điểm nhận diện: {row['detection_score']:.1%}")
                        else:
                            st.caption(f"Điểm nhận diện: {row['detection_score']:.1%} · Điểm phân loại: {row['quality_score']:.1%}")
                        if row['reason']:
                            st.caption(row['reason'])
        except Exception as exc:
            st.error(f'Chưa thể phân tích ảnh: {exc}')
    else:
        st.write('Chọn hoặc chụp ảnh; kết quả sẽ tự xuất hiện tại đây.')

from datetime import date
st.markdown(f'''<footer class="food-footer">
<strong>🥬 FoodQuality</strong>
<span>Nhận diện thực phẩm qua hình ảnh</span>
<span>© {date.today().year} FoodQuality</span>
</footer>''',unsafe_allow_html=True)

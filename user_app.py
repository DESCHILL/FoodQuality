"""End-user interface. Contains no dataset collection or training controls."""
import hashlib
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

st.set_page_config(page_title='FoodVision · Nhận diện thực phẩm', page_icon='🥬', layout='wide')
st.markdown('''<style>
.block-container{max-width:1150px;padding-top:2.5rem}
[data-testid="stAppViewContainer"]{background:#f7faf8}
[data-testid="stMain"]{color:#18392d}
[data-testid="stMain"] p,[data-testid="stMain"] label{color:#18392d}
h1,h2,h3{color:#164c3d}
[data-testid="stToolbar"]{display:none}
[data-testid="stMetric"]{background:white;border:1px solid #dce9e1;border-radius:12px;padding:18px}
</style>''', unsafe_allow_html=True)
st.caption('FOODVISION • NHẬN DIỆN QUA HÌNH ẢNH')
st.title('Hiểu rõ thực phẩm trong ảnh của bạn')
st.write('Tải ảnh hoặc chụp ảnh để xác định thực phẩm và quan sát tình trạng bên ngoài.')

ready = False
food9_mode=False
fusion_mode=False
consensus_mode=False
if (ROOT/'models/food9.json').exists():
    with st.expander('Nguồn mô hình'):
        options=['Demo 9 loại · dataset có nhãn','Minh họa huấn luyện Google']
        if (ROOT/'models/fusion.json').exists(): options.insert(0,'Kết hợp SIC + FoodQuality')
        if (ROOT/'models/consensus.json').exists(): options.insert(0,'Nhận diện từ cả 3 nguồn')
        selected=st.selectbox('Chế độ',options)
    fusion_mode=selected.startswith('Kết hợp')
    consensus_mode=selected.startswith('Nhận diện từ cả')
    food9_mode=selected.startswith('Demo') or fusion_mode or consensus_mode
    if food9_mode:
        BUNDLE=ROOT/'models/consensus.json' if consensus_mode else ROOT/'models/fusion.json' if fusion_mode else ROOT/'models/food9.json'
try:
    if food9_mode:
        import json
        registry=json.loads(BUNDLE.read_text(encoding='utf8'))
        weight_registry=json.loads(Path(registry['quality_registry']).read_text(encoding='utf8')) if fusion_mode or consensus_mode else registry
        if not Path(weight_registry['weights']).is_file():
            raise ValueError('Thiếu trọng số demo 9 loại.')
        bundle={'detector':{'classes':registry['foods']},'classifier':{}}
    else:
        bundle = load_bundle(BUNDLE)
    ready = True
    st.success('Hệ thống sẵn sàng phân tích ảnh.')
    st.caption('Phạm vi mô hình hiện tại: '+', '.join(FOODS.get(name,name) for name in bundle['detector']['classes']))
    if food9_mode:
        st.caption('Nguồn: '+weight_registry['source']+' · Suy luận cục bộ.')
        if fusion_mode:
            st.caption('Ưu tiên bộ phân loại đã học từ hai nguồn; YOLO SIC hỗ trợ tìm vùng khi nhận dạng chưa rõ. Chất lượng chỉ hỗ trợ 9 loại trên; nhãn khác chưa được đánh giá Tươi/Hỏng.')
        if consensus_mode:
            st.caption('FoodQuality, SIC EfficientNet và SIC YOLO đều chạy. Chọn tên có điểm cao nhất sau hiệu chỉnh; đánh giá Tươi/Hỏng riêng, có thể chưa xác định khi các mô hình bất đồng.')
        st.info('Đánh giá ngoại quan theo nhãn dataset; không xác nhận an toàn ăn uống. Ảnh không rõ sẽ trả Chưa xác định.')
        st.caption('Dùng ảnh một loại thực phẩm trong phạm vi trên. Riêng thịt cừu có ít nhóm ảnh độc lập; kết quả còn hạn chế. Không hỗ trợ thịt lợn trong bản demo 9 loại.')
    elif any(bundle[k].get('experimental') for k in ('detector','classifier')):
        st.warning('Bản thử nghiệm: mô hình học từ nhãn/vùng tự động chưa kiểm chứng. Kết quả chỉ để tham khảo ngoại quan, chưa có độ chính xác thực tế được xác nhận.')
except (ValueError, OSError, KeyError) as exc:
    st.info(str(exc))

@st.cache_resource
def get_predictor(bundle_fingerprint, implementation_fingerprint, registry_path, use_food9):
    import importlib
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

left,right=st.columns([1,1],gap='large')
with left:
    st.subheader('1. Chọn ảnh thực phẩm')
    mode=st.radio('Nguồn ảnh',['Tải ảnh','Chụp ảnh'],horizontal=True)
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
    st.caption('Ảnh được phân tích tự động ngay khi tải lên hoặc chụp xong.')
with right:
    st.subheader('2. Kết quả phân tích')
    if analyze:
        try:
            with st.spinner('Đang nhận diện và đánh giá...'):
                fingerprint=hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
                implementation_bytes=(ROOT/'foodquality8'/('food9.py' if food9_mode else 'inference.py')).read_bytes()
                if fusion_mode: implementation_bytes+=(ROOT/'foodquality8/fusion.py').read_bytes()
                if consensus_mode: implementation_bytes+=(ROOT/'foodquality8/consensus.py').read_bytes()
                if food9_mode or fusion_mode or consensus_mode:
                    implementation_bytes+=(ROOT/'foodquality8/food9.py').read_bytes()+(ROOT/'foodquality8/robust_views.py').read_bytes()+(ROOT/'foodquality8/quality_ensemble.py').read_bytes()
                implementation=hashlib.sha256(implementation_bytes).hexdigest()
                request_key=hashlib.sha256(uploaded.getvalue()).hexdigest()+fingerprint+implementation
                fresh_result=st.session_state.get('analysis_key')!=request_key
                if fresh_result:
                    predictor=get_predictor(fingerprint,implementation,str(BUNDLE),'consensus' if consensus_mode else 'fusion' if fusion_mode else food9_mode)
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
                        if row.get('prediction_source'): st.caption(row['prediction_source'])
                        if row.get('source_scores'):
                            with st.expander('Kết quả cả 3 bộ nhận diện',expanded=True):
                                for source in row['source_scores']:
                                    name=FOODS.get(source['food'],source['food']) if source['food'] else 'Không phát hiện'
                                    st.write(f"{source['source']}: {name} — {source['score']:.1%}")
                        if row.get('quality_evidence'):
                            with st.expander('Đối chiếu trạng thái'):
                                for item in row['quality_evidence']:
                                    label='Bất thường ngoại quan' if item['condition']=='defect' else CONDITIONS.get(item['condition'],item['condition'])
                                    st.write(f"{item['source']}: {FOODS.get(item['food'],item['food'])} · {label} — {item['score']:.1%}")
                        if row['reason']:
                            st.caption(row['reason'])
                        if row.get('visual_signs'):
                            with st.expander('Dấu hiệu ngoại quan được tham khảo'):
                                for sign in row['visual_signs']:
                                    st.write(sign)
                                st.caption('Đặc trưng màu/bề mặt được học cùng dataset; không dùng luật màu đơn lẻ để kết luận hỏng. Không quan sát được mùi hoặc vi sinh từ ảnh.')
                        if row['condition']=='unknown' and row.get('quality_candidates'):
                            st.write('Các nhãn mô hình đang cân nhắc — chưa phải kết luận:')
                            for candidate in row['quality_candidates']:
                                label=CONDITIONS.get(candidate['condition'],candidate['condition'])
                                name=FOODS.get(candidate['food'],candidate['food'])
                                st.caption(f"{name} · {label}: điểm {candidate['score']:.1%}")
                if fresh_result:
                    st.session_state.setdefault('history',[]).insert(0,[{**r,'food':FOODS.get(r['food'],r['food'])} for r in rows])
                    st.session_state['history']=st.session_state['history'][:10]
        except Exception as exc:
            st.error(f'Chưa thể phân tích ảnh: {exc}')
    else:
        st.write('Chọn hoặc chụp ảnh; kết quả sẽ tự xuất hiện tại đây.')
st.caption('Đánh giá ngoại quan không khẳng định an toàn ăn uống. Điểm mô hình chưa phải xác suất đã được hiệu chuẩn.')
with st.expander('Các thực phẩm được hỗ trợ'):
    if ready:
        st.write(', '.join(FOODS.get(name,name) for name in bundle['detector']['classes']))
    else:
        st.write('Danh sách sẽ xuất hiện sau khi mô hình được kích hoạt.')
with st.expander('Lịch sử trong phiên này'):
    history=st.session_state.get('history',[])
    if not history:
        st.write('Chưa có lượt phân tích.')
    for i,rows in enumerate(history,1):
        st.write(f'Lượt {i}',[dict(thuc_pham=r['food'],trang_thai=CONDITIONS.get(r['condition'],r['condition'])) for r in rows])
    if st.button('Xóa lịch sử'):
        st.session_state['history']=[]
        st.rerun()

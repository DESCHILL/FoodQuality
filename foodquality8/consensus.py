"""Three-source recognition with validation-calibrated scores and separate quality evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import ImageOps
from .food9 import Food9Predictor,FOODS

def select_recognition(sources):
    available=[s for s in sources if s.get('food') is not None]
    if not available:raise ValueError('Không có kết quả từ các mô hình.')
    return max(available,key=lambda s:s['score'])

def decide_quality(food,evidence,threshold=.8):
    usable=[e for e in evidence if e['food']==food and e['food_score']>=.6 and e['score']>=threshold]
    if not usable:return 'unknown','Chưa đủ bằng chứng về trạng thái.'
    conditions={e['condition'] for e in usable}
    if 'fresh' in conditions and conditions-{'fresh'}:
        return 'unknown','Các mô hình đánh giá trạng thái không thống nhất.'
    # Generic defects cannot establish spoilage, but do not contradict a separate spoiled prediction.
    return ('spoiled' if 'spoiled' in conditions else usable[0]['condition']),''

class ConsensusPredictor:
    def __init__(self,registry):
        from ultralytics import YOLO
        self.config=json.loads(Path(registry).read_text(encoding='utf8'))
        for key in ('quality_registry','sic_classifier','detector'):
            if hashlib.sha256(Path(self.config[key]).read_bytes()).hexdigest()!=self.config[key+'_sha256']:
                raise ValueError('Tài nguyên mô hình đã thay đổi: '+key)
        self.base=Food9Predictor(self.config['quality_registry'])
        self.sic=torch.load(self.config['sic_classifier'],weights_only=True,map_location='cpu')
        if not all(torch.equal(v,self.base.data['encoder'][k]) for k,v in self.sic['model_state_dict'].items() if not k.startswith('classifier.')):
            raise ValueError('Các bộ trích đặc trưng không tương thích.')
        self.names=self.sic['class_names']
        self.detector=YOLO(self.config['detector'])

    def predict(self,image):
        from .food9 import visual_features
        im=ImageOps.exif_transpose(image).convert('RGB')
        calibration=self.config['calibration']
        with torch.inference_mode():
            deep=self.base.encoder(self.base.transform(im).unsqueeze(0))[0]
            from .robust_views import LEGACY,extract_features
            raw=(torch.cat([deep,torch.from_numpy(visual_features(im))]) if self.base.preprocessing==LEGACY
                 else extract_features(self.base.encoder,[im],self.base.preprocessing)[0])
            x=((raw-self.base.data['mean'])/self.base.data['std']).clamp(-8,8)[None]
            fq_logits=self.base.food(x)[0]
            fq_prob=(fq_logits/calibration['foodquality_temperature']).softmax(0)
            fq_index=int(fq_prob.argmax());fq_food=FOODS[fq_index]
            q_logits=self.base.quality(x).reshape(9,2)[fq_index]
            fq_quality=(q_logits/calibration['foodquality_quality_temperature']).softmax(0)
            sic_logits=deep@self.sic['model_state_dict']['classifier.1.weight'].T+self.sic['model_state_dict']['classifier.1.bias']
            # Food logits marginalize over appearance classes before temperature scaling.
            sic_foods=sorted({n.split('__')[0] for n in self.names})
            food_logits=torch.stack([sic_logits[[i for i,n in enumerate(self.names) if n.split('__')[0]==f]].logsumexp(0) for f in sic_foods])
            sic_food_prob=(food_logits/calibration['sic_temperature']).softmax(0)
            sic_index=int(sic_food_prob.argmax());sic_food=sic_foods[sic_index]
            indices=[i for i,n in enumerate(self.names) if n.split('__')[0]==sic_food]
            sq=(sic_logits[indices]/calibration.get('sic_quality_by_food',{}).get(sic_food,calibration['sic_quality_temperature'])).softmax(0)
        sources=[dict(source='FoodQuality',food=fq_food,score=float(fq_prob[fq_index])),
            dict(source='SIC EfficientNet',food=sic_food,score=float(sic_food_prob[sic_index]))]
        result=self.detector.predict(im,conf=.15,imgsz=640,verbose=False,device='cpu',max_det=12)[0]
        yolo=[]
        for box in result.boxes:
            label=result.names[int(box.cls.item())].strip().lower().replace(' ','_')
            raw=float(box.conf.item())
            a,b=calibration['yolo_platt']
            z=a*np.log(max(raw,1e-6)/max(1-raw,1e-6))+b
            score=float(1/(1+np.exp(-np.clip(z,-40,40))))
            yolo.append(dict(source='SIC YOLO',food=label,score=score,raw_score=raw,bbox=box.xyxy[0].tolist()))
        sources.append(max(yolo,key=lambda s:s['score']) if yolo else dict(source='SIC YOLO',food=None,score=0.))
        winner=select_recognition(sources);food=winner['food']
        evidence=[dict(source='FoodQuality',food=fq_food,food_score=float(fq_prob[fq_index]),
            condition=('fresh','spoiled')[int(fq_quality.argmax())],score=float(fq_quality.max()))]
        sic_label=self.names[indices[int(sq.argmax())]].split('__')[1]
        # A generic defect is not proof of spoilage; only normal appearance supports fresh.
        if len(indices)==2:
            evidence.append(dict(source='SIC EfficientNet',food=sic_food,food_score=float(sic_food_prob[sic_index]),
                condition='fresh' if sic_label=='normal_appearance' else 'defect',score=float(sq.max())))
        condition,reason=decide_quality(food,evidence,self.config.get('quality_threshold',.8))
        if condition=='defect':condition='unknown';reason='SIC phát hiện bất thường ngoại quan; chưa đủ để xác nhận hỏng.'
        if food not in FOODS:condition='unknown';reason='Thực phẩm này chưa thuộc phạm vi đánh giá Tươi/Hỏng đã kiểm tra.'
        if winner['score']<.6:condition='unknown';reason='Điểm nhận dạng còn thấp.'
        matching=[e for e in evidence if e['food']==food]
        quality_score=max((e['score'] for e in matching),default=0.)
        return [dict(food=food,condition=condition,detection_score=winner['score'],quality_score=quality_score,
            bbox=winner.get('bbox'),tentative=winner['score']<.6,reason=reason,
            quality_candidates=[],visual_signs=[],source_scores=sources,quality_evidence=evidence,
            prediction_source='Đã chạy 3 bộ nhận diện; chọn điểm cao nhất sau hiệu chỉnh trên validation')]

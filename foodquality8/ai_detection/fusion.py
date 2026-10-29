"""SIC object localization + source-labelled Food9 appearance classification."""
import hashlib
import json
import math
from pathlib import Path
from PIL import ImageOps
from foodquality8.ai_quality.food9 import Food9Predictor, FOODS

def canonical_food(label):
    label=label.strip().lower().replace('_',' ').replace('-',' ')
    return label if label in FOODS else None

def reconcile(row, detected_food, confidence, box, crop_size):
    result=dict(row)
    result['bbox']=box
    result['detection_score']=confidence
    result['prediction_source']='SIC YOLO + FoodQuality; suy luận cục bộ'
    if detected_food != row['food']:
        result.update(food=detected_food,condition='unknown',quality_candidates=[],tentative=True,
            reason='Mô hình khoanh vùng và mô hình chất lượng không thống nhất loại thực phẩm.')
    elif min(crop_size)<48:
        result.update(condition='unknown',reason='Vùng thực phẩm quá nhỏ; hãy chụp gần hơn.')
    return result

class FusionPredictor:
    def __init__(self,registry):
        from ultralytics import YOLO
        self.config=json.loads(Path(registry).read_text(encoding='utf8'))
        detector=Path(self.config['detector'])
        if hashlib.sha256(detector.read_bytes()).hexdigest()!=self.config['detector_sha256']:
            raise ValueError('Trọng số SIC đã thay đổi; cần kiểm tra lại.')
        quality=Path(self.config['quality_registry'])
        if hashlib.sha256(quality.read_bytes()).hexdigest()!=self.config['quality_registry_sha256']:
            raise ValueError('Cấu hình mô hình chất lượng đã thay đổi; cần đánh giá lại bộ kết hợp.')
        self.quality=Food9Predictor(quality)
        self.detector=YOLO(str(detector))
        if self.detector.task!='detect': raise ValueError('Checkpoint SIC không phải mô hình khoanh vùng.')

    def predict(self,image):
        im=ImageOps.exif_transpose(image).convert('RGB')
        whole=self.quality.predict(im)
        if self.config.get('routing')=='classification_first' and not whole[0].get('tentative'):
            whole[0]['prediction_source']='FoodQuality học từ FoodClass + AgML; đủ điểm nhận dạng toàn ảnh'
            return whole
        detections=self.detector.predict(im,conf=.4,iou=.5,imgsz=640,max_det=12,verbose=False,device='cpu')[0]
        results=[]
        for detected in detections.boxes:
            coords=detected.xyxy[0].tolist()
            if not all(math.isfinite(v) for v in coords):continue
            x1,y1,x2,y2=coords
            box=[max(0,int(x1)),max(0,int(y1)),min(im.width,math.ceil(x2)),min(im.height,math.ceil(y2))]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            label=detections.names[int(detected.cls.item())]
            food=canonical_food(label);score=float(detected.conf.item())
            if food is None:
                results.append(dict(food=label,condition='unknown',quality_score=0.,detection_score=score,
                    bbox=box,tentative=False,quality_candidates=[],visual_signs=[],
                    reason='Nhận diện được nhãn này nhưng chưa có mô hình Tươi/Hỏng phù hợp. Không suy trạng thái từ tên món.',
                    prediction_source='SIC YOLO; chất lượng ngoài phạm vi'))
                continue
            # Small padding retains edges/mold; clamp to the original image.
            pad=int(min(box[2]-box[0],box[3]-box[1])*.08)
            crop=im.crop((max(0,box[0]-pad),max(0,box[1]-pad),min(im.width,box[2]+pad),min(im.height,box[3]+pad)))
            use_whole=(len(detections.boxes)==1 and self.config.get('single_object_quality')=='whole')
            row=self.quality.predict(im if use_whole else crop)[0]
            results.append(reconcile(row,food,score,box,crop.size))
        if results:return results
        # SIC has no raw beef/mutton labels. Full-image classification remains available.
        result=whole
        for row in result:
            row['prediction_source']='FoodQuality toàn ảnh; SIC không tìm được vùng đủ điểm'
            row['reason']=('Không tìm được vùng bằng YOLO; đánh giá toàn ảnh, phù hợp ảnh một thực phẩm. '+row['reason']).strip()
        return result

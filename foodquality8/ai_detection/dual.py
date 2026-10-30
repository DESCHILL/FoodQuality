"""Two-checkpoint inference: YOLO localization and conditional visual quality."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps

from foodquality8.ai_quality.food9 import Food9Predictor, FOODS
from foodquality8.ai_quality.robust_views import extract_features, LEGACY
from foodquality8.ai_quality.quality_ensemble import condition_logits
from foodquality8.ai_detection.consensus import select_recognition, decide_quality


def combine(sources, evidence, threshold):
    winner = select_recognition(sources)
    food = winner['food']
    condition, reason = decide_quality(food, evidence, threshold)
    if food not in FOODS or winner['score'] < .6:
        condition, reason = 'unknown', 'Chưa đủ bằng chứng nhận diện trong phạm vi hỗ trợ.'
    matching = [e for e in evidence if e['food'] == food]
    return dict(food=food, condition=condition, detection_score=winner['score'],
                quality_score=max((e['score'] for e in matching), default=0.),
                bbox=winner.get('bbox'), tentative=winner['score'] < .6, reason=reason,
                quality_candidates=[], visual_signs=[], source_scores=sources,
                quality_evidence=evidence, prediction_source='YOLO + bộ phân loại chất lượng')


class DualPredictor:
    def __init__(self, registry):
        from ultralytics import YOLO
        self.config = json.loads(Path(registry).read_text(encoding='utf8'))
        for key in ('quality_registry', 'detector'):
            if hashlib.sha256(Path(self.config[key]).read_bytes()).hexdigest() != self.config[key+'_sha256']:
                raise ValueError('Checkpoint/configuration checksum mismatch: '+key)
        self.base = Food9Predictor(self.config['quality_registry'])
        self.detector = YOLO(self.config['detector'])

    def predict(self, image):
        return self.predict_cached(image)

    def predict_cached(self, image, detail=None, legacy=None):
        image = ImageOps.exif_transpose(image).convert('RGB')
        calibration = self.config['calibration']
        with torch.inference_mode():
            if detail is None:
                detail = extract_features(self.base.encoder, [image], self.base.preprocessing)[0]
                legacy = extract_features(self.base.encoder, [image], LEGACY)[0]
            raw, old = torch.as_tensor(detail), torch.as_tensor(legacy)
            if raw.shape != (1331,) or old.shape != (1331,) or not torch.isfinite(raw).all() or not torch.isfinite(old).all():
                raise ValueError('Invalid feature vectors')
            x = ((raw-self.base.data['mean'])/self.base.data['std']).clamp(-8,8)[None]
            probability = (self.base.food(x)[0]/calibration['foodquality_temperature']).softmax(0)
            index = int(probability.argmax())
            quality = (condition_logits(self.base.data, raw[None], old[None])[0,index]/calibration['foodquality_quality_temperature']).softmax(0)
        sources = [dict(source='FoodQuality', food=FOODS[index], score=float(probability[index]))]
        result = self.detector.predict(image, conf=.15, imgsz=640, device='cpu', verbose=False, max_det=12)[0]
        detections = []
        for box in result.boxes:
            score = float(box.conf.item())
            a, b = calibration['yolo_platt']
            z = a*np.log(max(score,1e-6)/max(1-score,1e-6))+b
            detections.append(dict(source='SIC YOLO', food=result.names[int(box.cls.item())].strip().lower().replace(' ','_'),
                                   score=float(1/(1+np.exp(-np.clip(z,-40,40)))), raw_score=score, bbox=box.xyxy[0].tolist()))
        sources.append(max(detections,key=lambda d:d['score']) if detections else dict(source='SIC YOLO', food=None, score=0.))
        evidence = [dict(source='FoodQuality', food=FOODS[index], food_score=float(probability[index]),
                         condition=('fresh','spoiled')[int(quality.argmax())], score=float(quality.max()))]
        return [combine(sources,evidence,self.config.get('quality_threshold',.8))]

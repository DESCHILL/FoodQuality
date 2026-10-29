"""Shared ImageNet features, food recognition and food-conditional binary freshness."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

FOODS=['apple','banana','beef','carrot','chicken','cucumber','mutton','orange','tomato']
NAMES=dict(apple='Táo',banana='Chuối',beef='Thịt bò',carrot='Cà rốt',chicken='Thịt gà',cucumber='Dưa chuột',mutton='Thịt cừu',orange='Cam',tomato='Cà chua')


def visual_features(image):
    """Descriptive appearance features, never hard-coded spoilage labels."""
    import cv2
    rgb=np.asarray(image.convert('RGB').resize((128,128)))
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    parts=[]
    for i,end in enumerate((180,256,256)):
        hist=np.histogram(hsv[:,:,i],bins=16,range=(0,end))[0].astype(np.float32)
        parts.extend(hist/hist.sum())
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    parts.extend([float((gray<45).mean()),float(((hsv[:,:,1]<35)&(hsv[:,:,2]>200)).mean()),float(gray.std()/255)])
    return np.asarray(parts,dtype=np.float32)


def encoder():
    import torch
    from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
    torch.set_num_threads(min(4,torch.get_num_threads()))
    weights=EfficientNet_B0_Weights.DEFAULT
    model=efficientnet_b0(weights=weights)
    model.classifier=torch.nn.Identity()
    model.eval()
    return model,weights.transforms()


class Food9Predictor:
    def __init__(self,registry):
        import torch
        from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
        self.registry=json.loads(Path(registry).read_text(encoding='utf8'))
        path=Path(self.registry['weights'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.registry['sha256']:
            raise ValueError('Trọng số Food9 đã thay đổi.')
        self.data=torch.load(path,map_location='cpu',weights_only=True)
        torch.set_num_threads(min(4,torch.get_num_threads()))
        self.encoder=efficientnet_b0(weights=None)
        self.encoder.classifier=torch.nn.Identity()
        self.encoder.load_state_dict(self.data['encoder'])
        self.encoder.eval()
        self.transform=EfficientNet_B0_Weights.DEFAULT.transforms()
        from foodquality8.ai_quality.robust_views import LEGACY,validate_recipe
        self.preprocessing=validate_recipe(self.data.get('preprocessing',LEGACY))
        dim=len(self.data['mean'])
        self.food=torch.nn.Linear(dim,len(FOODS))
        self.quality=torch.nn.Linear(dim,len(FOODS)*2)
        self.food.load_state_dict(self.data['food_head'])
        self.quality.load_state_dict(self.data['quality_head'])
        self.food.eval();self.quality.eval()

    def predict(self,image):
        import torch
        im=ImageOps.exif_transpose(image).convert('RGB')
        with torch.inference_mode():
            from foodquality8.ai_quality.robust_views import extract_features
            raw=extract_features(self.encoder,[im],self.preprocessing)[0]
            x=((raw-self.data['mean'])/self.data['std']).clamp(-8,8).unsqueeze(0)
            food_prob=self.food(x).softmax(1)[0]
            i=int(food_prob.argmax())
            from foodquality8.ai_quality.quality_ensemble import condition_logits
            legacy_raw=None
            if self.data.get('legacy_quality'):
                from foodquality8.ai_quality.robust_views import LEGACY
                legacy_raw=extract_features(self.encoder,[im],LEGACY)
            condition_prob=condition_logits(self.data,raw[None],legacy_raw)[0,i].softmax(0)
        food=FOODS[i]
        score=float(food_prob[i])
        quality_score=float(condition_prob.max())
        confident=score>=.65 and quality_score>=.65
        condition=('fresh','spoiled')[int(condition_prob.argmax())] if confident else 'unknown'
        visual=visual_features(im)
        signs=[f'Vùng tối trong toàn ảnh: {visual[-3]:.1%}; vùng sáng ít màu: {visual[-2]:.1%}.']
        if food in ('beef','chicken','mutton'):
            signs.append('Vùng trắng có thể là mỡ/da hoặc phản sáng; màu sẫm có thể do ánh sáng/oxy hóa. Không dùng riêng các dấu hiệu này để kết luận hỏng.')
        else:
            signs.append('Vết thâm có thể do dập/chín; nền ảnh cũng ảnh hưởng các tỷ lệ màu. Cần phân biệt với mốc hoặc tổn thương lan rộng.')
        return [dict(food=food,condition=condition,quality_score=quality_score,detection_score=score,
            bbox=None,tentative=score<.65,reason='' if confident else 'Hai mô hình chưa đủ chắc chắn; không ép kết luận tươi/hỏng.',
            visual_signs=signs,quality_candidates=[dict(food=food,condition=c,score=float(condition_prob[j])) for j,c in enumerate(('fresh','spoiled'))],
            prediction_source='FoodClass · mô hình cục bộ nhận dạng và chất lượng')]

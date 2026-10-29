"""Shared two-model inference; never falls back to fabricated predictions."""
import hashlib
import json
from pathlib import Path

from PIL import ImageOps
from foodquality8.dataset.model_data import write_json


def activate_run(run, bundle_path, allow_experimental=False):
    run, bundle_path = Path(run).resolve(), Path(bundle_path).resolve()
    data = json.loads((run / 'run.json').read_text(encoding='utf8'))
    if data.get('status') != 'complete' or not (run/'test-metrics.json').is_file():
        raise ValueError('Chi kich hoat luot train hoan tat va co danh gia test.')
    experimental=data.get('dataset',{}).get('label_mode') in ('google_query_suggested','automatic_suggested')
    if experimental and not allow_experimental:
        raise ValueError('Model này dùng nhãn gợi ý Google, chỉ để thử nghiệm trong FoodQuality. Cần dữ liệu kiểm chứng trước khi dùng cho người dùng.')
    kind = data['kind']
    if kind not in ('classifier', 'detector'):
        raise ValueError('Loai model khong hop le.')
    weights = Path(data['weights']).resolve()
    if not weights.is_relative_to(run) or not weights.is_file():
        raise ValueError('Khong tim thay trong so trong thu muc train.')
    bundle = json.loads(bundle_path.read_text(encoding='utf8')) if bundle_path.exists() else {}
    entry = dict(weights=str(weights), sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                 run=str(run), classes=data['dataset']['classes'], class_labels=data['dataset']['class_labels'],experimental=experimental)
    candidate = {**bundle, kind: entry}
    if 'detector' in candidate and 'classifier' in candidate:
        detector_foods = set(candidate['detector']['classes'])
        quality_foods = {x['food'] for x in candidate['classifier']['class_labels']}
        if not detector_foods.issubset(quality_foods):
            raise ValueError('Classifier chua ho tro cac thuc pham cua detector: ' + ', '.join(sorted(detector_foods-quality_foods)))
    write_json(bundle_path, candidate)
    return candidate


def load_bundle(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Hệ thống chưa có mô hình được kích hoạt. Vui lòng quay lại sau khi quản trị viên hoàn tất huấn luyện.')
    bundle = json.loads(path.read_text(encoding='utf8'))
    for kind in ('detector', 'classifier'):
        if kind not in bundle:
            raise ValueError('Hệ thống đang chuẩn bị mô hình nhận diện và đánh giá. Chưa thể phân tích ảnh.')
        weights = Path(bundle[kind]['weights'])
        if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != bundle[kind]['sha256']:
            raise ValueError('Mô hình đã thay đổi hoặc bị thiếu; cần quản trị viên kiểm tra lại.')
    return bundle


def quality_decision(probabilities, labels, food, threshold):
    index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    confidence = float(probabilities[index])
    label = labels[index]
    if label['food'] != food or confidence < threshold:
        return dict(condition='unknown', quality_score=confidence, reason='Chưa đủ bằng chứng hoặc hai mô hình không thống nhất.')
    return dict(condition=label['condition'], quality_score=confidence, reason='')


def select_detections(boxes, threshold=.35, tentative_threshold=.15):
    confident=[b for b in boxes if float(b.conf.item())>=threshold]
    if confident:
        return confident,False
    tentative=[b for b in boxes if float(b.conf.item())>=tentative_threshold]
    return ([max(tentative,key=lambda b:float(b.conf.item()))],True) if tentative else ([],False)


class FoodPredictor:
    def __init__(self, bundle_path):
        import torch
        from torchvision.models import efficientnet_b0
        from torchvision import transforms
        from ultralytics import YOLO
        self.bundle = load_bundle(bundle_path)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.detector = YOLO(self.bundle['detector']['weights'])
        checkpoint = torch.load(self.bundle['classifier']['weights'], map_location='cpu', weights_only=True)
        if checkpoint.get('architecture') != 'efficientnet_b0':
            raise ValueError('Classifier khong phai EfficientNet-B0.')
        self.labels = checkpoint['class_labels']
        if self.labels != self.bundle['classifier']['class_labels']:
            raise ValueError('Metadata lop khong khop checkpoint.')
        detector_names = self.detector.names
        if isinstance(detector_names, dict):
            detector_names = [detector_names[i] for i in range(len(detector_names))]
        if detector_names != self.bundle['detector']['classes']:
            raise ValueError('Ten lop detector khong khop bundle.')
        self.classifier = efficientnet_b0(weights=None)
        self.classifier.classifier[1] = torch.nn.Linear(self.classifier.classifier[1].in_features, len(self.labels))
        self.classifier.load_state_dict(checkpoint['state_dict'])
        self.classifier.eval()
        prep = checkpoint['preprocess']
        self.transform = quality_transform(prep)

    def predict(self, image, detection_threshold=.35, quality_threshold=.6):
        import torch
        im = ImageOps.exif_transpose(image).convert('RGB')
        if im.width * im.height > 25_000_000:
            raise ValueError('Ảnh quá lớn, vui lòng chọn ảnh dưới 25 megapixel.')
        # Keep a lower-confidence candidate visible, but never present it as a confirmed detection.
        tentative_threshold=min(.15,detection_threshold)
        result = self.detector.predict(im, conf=tentative_threshold, device='cpu', verbose=False, max_det=20)[0]
        boxes,tentative=select_detections(result.boxes,detection_threshold,tentative_threshold)
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            food = result.names[int(box.cls.item())]
            crop = im.crop((x1, y1, x2, y2))
            with torch.inference_mode():
                probabilities = self.classifier(self.transform(crop).unsqueeze(0)).softmax(1)[0].tolist()
            decision = quality_decision(probabilities, self.labels, food, quality_threshold)
            if tentative:
                decision.update(condition='unknown',reason='Vùng nghi ngờ: điểm nhận diện dưới ngưỡng. Chưa xác nhận loại thực phẩm hoặc chất lượng.')
            candidates=[dict(**self.labels[i],score=float(probabilities[i])) for i in sorted(range(len(probabilities)),key=lambda i:probabilities[i],reverse=True)[:3]]
            detections.append(dict(food=food, detection_score=float(box.conf.item()), bbox=[x1,y1,x2,y2],
                tentative=tentative,quality_candidates=candidates, **decision))
        return detections


def quality_transform(prep):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(prep['resize'], interpolation=transforms.InterpolationMode(prep.get('interpolation','bicubic')),
                          antialias=prep.get('antialias',True)),
        transforms.CenterCrop(prep['crop']), transforms.ToTensor(), transforms.Normalize(prep['mean'],prep['std'])])

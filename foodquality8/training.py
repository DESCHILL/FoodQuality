"""Standalone training jobs; GUI launches these in a separate process."""
import argparse
import json
import random
import time
from pathlib import Path

from .model_data import check_release, write_json


def choose_device(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('PyTorch hien tai khong dung duoc CUDA. Chon CPU hoac cai ban CUDA phu hop.')
    return requested


def evaluate_classifier(model, loader, device, classes):
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    true, predicted = [], []
    loss, count = 0., 0
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += float(torch.nn.functional.cross_entropy(logits, y, reduction='sum').item())
            count += len(y)
            true.extend(y.cpu().tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    report = classification_report(true, predicted, labels=list(range(len(classes))),
                                   target_names=classes, zero_division=0, output_dict=True)
    return dict(loss=loss / max(1, count), accuracy=report['accuracy'], macro_f1=report['macro avg']['f1-score'],
                per_class=report, confusion_matrix=confusion_matrix(true, predicted, labels=list(range(len(classes)))).tolist(),
                classes=classes, samples=count)


def train_classifier(info, run, epochs, batch, device, stop, pretrained=True):
    run, stop = Path(run).resolve(), Path(stop).resolve()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(42)
    random.seed(42)
    weights = EfficientNet_B0_Weights.DEFAULT
    eval_transform = weights.transforms()
    train_transform = transforms.Compose([transforms.Resize(256,interpolation=eval_transform.interpolation), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std)])
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = datasets.ImageFolder(Path(info['root']) / 'classification' / split,
                                  transform=train_transform if split == 'train' else eval_transform)
        if ds.classes != info['classes']:
            raise ValueError('Thu tu lop giua cac tap khong khop.')
        loaders[split] = DataLoader(ds, batch_size=batch, shuffle=split == 'train', num_workers=0,
                                   generator=torch.Generator().manual_seed(42))
    model = efficientnet_b0(weights=weights if pretrained else None)
    # Fast transfer-learning baseline: frozen backbone, train only classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(info['classes']))
    model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    counts=torch.bincount(torch.tensor(loaders['train'].dataset.targets),minlength=len(info['classes'])).float()
    class_weights=(counts.sum()/counts.clamp_min(1)).sqrt()
    class_weights=(class_weights/class_weights.mean()).to(device)
    best, best_loss, stale = -1., float('inf'), 0
    history = []
    for epoch in range(epochs):
        if stop.exists():
            return False
        if epoch==2:
            # Adapt the final visual blocks gently; keep batch normalization frozen for small batches.
            for parameter in model.features[-2:].parameters():
                parameter.requires_grad=True
            optimizer=torch.optim.AdamW([
                {'params':model.features[-2:].parameters(),'lr':2e-5},
                {'params':model.classifier.parameters(),'lr':5e-4}],weight_decay=1e-3)
        model.eval()  # Keep frozen backbone batchnorm statistics unchanged.
        model.classifier.train()
        loss_total, count = 0., 0
        for x, y in loaders['train']:
            if stop.exists():
                return False
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y,weight=class_weights,label_smoothing=.05)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(y)
            count += len(y)
        metrics = evaluate_classifier(model, loaders['val'], device, info['classes'])
        history.append(dict(epoch=epoch+1, train_loss=loss_total/max(1, count), val_loss=metrics['loss'],
                            val_macro_f1=metrics['macro_f1'], val_accuracy=metrics['accuracy']))
        print(json.dumps(history[-1]), flush=True)
        write_json(run / 'history.json', history)
        if metrics['macro_f1'] > best+1e-8 or (abs(metrics['macro_f1']-best)<=1e-8 and metrics['loss']<best_loss):
            best, best_loss, stale = metrics['macro_f1'], metrics['loss'], 0
            checkpoint = dict(architecture='efficientnet_b0', state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                classes=info['classes'], class_labels=info['class_labels'],
                preprocess=dict(resize=256, crop=224, mean=eval_transform.mean, std=eval_transform.std,
                                interpolation=eval_transform.interpolation.value, antialias=True),
                epoch=epoch+1, validation=metrics, dataset_sha256=info['dataset_sha256'], pretrained=pretrained,
                training_recipe='warmup_then_last_two_blocks_balanced_v2')
            temp = run / 'best.tmp'
            torch.save(checkpoint, temp)
            temp.replace(run / 'best.pt')
            write_json(run / 'validation.json', metrics)
        else:
            stale += 1
        if stale >= 6:
            print('Tự dừng: kết quả kiểm tra không cải thiện qua 6 vòng.', flush=True)
            break
    if stop.exists():
        return False
    checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    write_json(run / 'test-metrics.json', evaluate_classifier(model, loaders['test'], device, info['classes']))
    return True


def train_detector(info, run, epochs, batch, device, stop, weights='yolo26n.pt', imgsz=640):
    run, stop = Path(run).resolve(), Path(stop).resolve()
    import yaml
    from ultralytics import YOLO
    model = YOLO(weights)
    def request_stop(trainer):
        if stop.exists():
            trainer.stop = True
    model.add_callback('on_train_epoch_end', request_stop)
    # Rewrite path locally so moving a release does not point to a different dataset.
    data_path = run / 'data.yaml'
    data_path.write_text(yaml.safe_dump(dict(path=str(Path(info['root'])/'detection'),
        train='images/train', val='images/val', test='images/test', names=info['classes'])), encoding='utf8')
    small=sum(info['counts']['train'].values())<200
    model.train(data=str(data_path), epochs=epochs, batch=batch, imgsz=imgsz, device=device,
                workers=0, project=str(run), name='yolo', exist_ok=False, patience=8, seed=42, plots=True,
                freeze=10 if small else None,mosaic=.3 if small else 1.,close_mosaic=min(5,epochs),
                hsv_h=0.,hsv_s=.1,hsv_v=.1,scale=.2,translate=.05,cos_lr=True)
    if stop.exists():
        return False
    best = run / 'yolo/weights/best.pt'
    if not best.is_file():
        raise RuntimeError('YOLO khong tao best.pt.')
    model = YOLO(best)
    validation=model.val(data=str(data_path),split='val',device=device,workers=0,plots=False,imgsz=imgsz,
                         project=str(run),name='validation')
    write_json(run/'validation.json',dict(map50=float(validation.box.map50),map50_95=float(validation.box.map),
        precision=float(validation.box.mp),recall=float(validation.box.mr),classes=info['classes']))
    metrics = model.val(data=str(data_path), split='test', device=device, workers=0, plots=True, imgsz=imgsz,
                        project=str(run), name='test')
    write_json(run / 'test-metrics.json', dict(map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
        precision=float(metrics.box.mp), recall=float(metrics.box.mr), classes=info['classes']))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['classifier', 'detector'])
    parser.add_argument('--release', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--experimental', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 500 or not 1 <= args.batch <= 128:
        parser.error('epochs: 1..500; batch: 1..128')
    run = Path(args.run).resolve()
    created = False
    try:
        info = check_release(args.release, args.kind, experimental=args.experimental)
        if info.get('label_mode') in ('google_query_suggested','automatic_suggested'):
            print('THỬ NGHIỆM: nhãn Google chưa kiểm chứng; điểm test chỉ đo mức khớp nhãn gợi ý, không phải độ chính xác chất lượng thực phẩm.',flush=True)
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        if args.check:
            return
        run.mkdir(parents=True, exist_ok=False)
        created = True
        device = choose_device(args.device)
        config = dict(kind=args.kind, status='running', device=device, epochs=args.epochs, batch=args.batch,
                      created_at=time.time(), dataset=info)
        write_json(run / 'run.json', config)
        print(f'Training {args.kind} on {device}; output: {run}', flush=True)
        function = train_classifier if args.kind == 'classifier' else train_detector
        completed = function(info, run, args.epochs, args.batch, device, run/'STOP')
        config['status'] = 'complete' if completed else 'stopped'
        config['weights'] = str(run / ('best.pt' if args.kind == 'classifier' else 'yolo/weights/best.pt'))
        write_json(run / 'run.json', config)
        if completed:
            from .model_report import make_report
            make_report(run)
        print('Hoan thanh va da danh gia test.' if completed else 'Da dung; khong kich hoat model chua hoan tat.', flush=True)
    except Exception as exc:
        if created and (run / 'run.json').is_file():
            current = json.loads((run/'run.json').read_text(encoding='utf8'))
            if current.get('status') != 'stopped':
                current.update(status='failed', error=str(exc))
                write_json(run/'run.json', current)
        parser.exit(1, f'Loi: {exc}\n')


if __name__ == '__main__':
    main()

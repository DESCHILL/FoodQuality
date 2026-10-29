"""YOLO training and evaluation."""
from pathlib import Path
from foodquality8.dataset.model_data import write_json

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


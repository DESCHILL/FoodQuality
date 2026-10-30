"""Measure external models on the project's fixed validation images."""
import argparse,json,hashlib,time,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision import models,transforms
from ultralytics import YOLO
from sklearn.metrics import classification_report,confusion_matrix

def main():
 p=argparse.ArgumentParser();p.add_argument('--external',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--food');a=p.parse_args()
 torch.set_num_threads(4)
 rows=[r for r in json.loads(a.manifest.read_text(encoding='utf8')) if r['split']=='val' and (not a.food or r['food']==a.food)]
 raw_path=a.external/'runs/raw_train_30/weights/best.pt';meat_path=a.external/'runs/efficientnet_meat_v1/best.pt'
 detector=YOLO(raw_path);ck=torch.load(meat_path,weights_only=True,map_location='cpu');classifier=models.efficientnet_b0(weights=None);classifier.classifier[1]=torch.nn.Linear(classifier.classifier[1].in_features,len(ck['classes']));classifier.load_state_dict(ck['model_state_dict']);classifier.eval()
 transform=transforms.Compose([transforms.Resize((ck['image_size'],ck['image_size'])),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 records=[];start=time.perf_counter()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 for i,r in enumerate(rows):
  with Image.open(r['path']) as opened:
   im=ImageOps.exif_transpose(opened).convert('RGB')
   result=detector.predict(im,conf=.25,imgsz=416,device='cpu',verbose=False)[0]
   boxes=sorted(result.boxes,key=lambda b:float(b.conf.item()),reverse=True)
   food=result.names[int(boxes[0].cls.item())].removeprefix('raw_') if boxes else 'unknown'
   record=dict(sha256=r['sha256'],food=r['food'],condition=r['condition'],source=r['source'],raw_prediction=food,raw_score=float(boxes[0].conf.item()) if boxes else 0.)
   if r['food'] in ('beef','chicken','mutton'):
    with torch.inference_mode():prob=classifier(transform(im)[None]).softmax(1)[0]
    label=ck['classes'][int(prob.argmax())];record.update(meat_prediction={'Fresh':'fresh','Spoiled':'spoiled'}.get(label,'unknown'),meat_score=float(prob.max()),meat_label=label)
   record['crop_prediction']='unknown'
   if boxes and food in ('beef','pork'):
    x1,y1,x2,y2=[int(v) for v in boxes[0].xyxy[0].tolist()];x1,y1=max(0,x1),max(0,y1);x2,y2=min(im.width,x2),min(im.height,y2)
    if x2>x1 and y2>y1:
     with torch.inference_mode():crop_prob=classifier(transform(im.crop((x1,y1,x2,y2)))[None]).softmax(1)[0]
     record['crop_prediction']={'Fresh':'fresh','Spoiled':'spoiled'}.get(ck['classes'][int(crop_prob.argmax())],'unknown')
   records.append(record)
  if (i+1)%100==0:print(f'{i+1}/{len(rows)}',flush=True)
 meat=[r for r in records if 'meat_prediction'in r]
 labels=['beef','chicken','mutton']
 report=dict(samples=len(rows),split='val',scope='Image-level comparison; no ground-truth boxes; not a localization mAP evaluation.',seconds=time.perf_counter()-start,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),weights={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw_path,meat_path]},raw_all_accuracy=float(np.mean([r['food']==r['raw_prediction'] for r in records])),raw_by_food={f:dict(samples=sum(r['food']==f for r in records),accuracy=float(np.mean([r['food']==r['raw_prediction'] for r in records if r['food']==f]))) for f in sorted({r['food'] for r in records})},meat_report=classification_report([r['condition'] for r in meat],[r['meat_prediction'] for r in meat],labels=['fresh','spoiled','unknown'],output_dict=True,zero_division=0),meat_by_food={f:dict(samples=sum(r['food']==f for r in meat),accuracy=float(np.mean([r['condition']==r['meat_prediction'] for r in meat if r['food']==f]))) for f in labels if any(r['food']==f for r in meat)},crop_pipeline_joint_accuracy=float(np.mean([r['raw_prediction']==r['food'] and r['crop_prediction']==r['condition'] for r in records])),records=records)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

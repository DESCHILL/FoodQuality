"""Train a reproducible 9-food demo; source labels are not claimed to be expert-verified."""
import hashlib
import json
import random
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from sklearn.metrics import accuracy_score,f1_score,classification_report
from foodquality8.core import phash
from foodquality8.food9 import FOODS,encoder,visual_features
from foodquality8.model_data import write_json


def prepare(base):
    rows=json.loads((base/'catalog.json').read_text(encoding='utf8'))
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    buckets=defaultdict(list); hashes=[]; bad=set()
    for i,row in enumerate(rows):
        try:
            with Image.open(row['path']) as image:
                h=int(phash(ImageOps.exif_transpose(image).convert('RGB')),16)
        except Exception:
            bad.add(i); hashes.append(0);continue
        hashes.append(h)
        candidates=set()
        for part in range(5):
            candidates.update(buckets[(part,(h>>(part*13))&8191)])
        for j in candidates:
            if (h^hashes[j]).bit_count()<=4:
                parents[find(i)]=find(j)
        for part in range(5): buckets[(part,(h>>(part*13))&8191)].append(i)
    groups=defaultdict(list)
    for i in range(len(rows)):
        if i not in bad: groups[find(i)].append(i)
    kept=[];conflicts=0
    for group in groups.values():
        if len({(rows[i]['food'],rows[i]['condition']) for i in group})>1:
            conflicts+=len(group);continue
        kept.append(dict(rows[group[0]],duplicate_group=rows[group[0]]['sha256']))
    classes=defaultdict(list)
    for row in kept: classes[(row['food'],row['condition'])].append(row)
    write_json(base/'group-counts.json',{f'{a}_{b}':len(v) for (a,b),v in classes.items()})
    result=[]
    for food in FOODS:
        for condition in ('fresh','spoiled'):
            items=sorted(classes[(food,condition)],key=lambda r:r['sha256'])
            if len(items)<10: raise ValueError(f'Thiếu ảnh độc lập: {food}/{condition}: {len(items)}')
            random.Random(42).shuffle(items)
            n=max(3,round(len(items)*.15))
            for i,row in enumerate(items):
                row['split']='test' if i<n else 'val' if i<2*n else 'train'
                result.append(row)
    write_json(base/'split.json',result)
    write_json(base/'data-report.json',dict(input=len(rows),retained=len(result),conflicting_images=conflicts,
        split_counts=dict(Counter(r['split'] for r in result)),class_counts={f'{a}_{b}':len(v) for (a,b),v in classes.items()},
        small_classes={f'{a}_{b}':len(v) for (a,b),v in classes.items() if len(v)<50},
        limitation='Demo only. Mutton has very few independent groups; its metrics are unstable. Exact/near-duplicate grouping; original specimen/session IDs unavailable, residual leakage remains possible.'))
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=ROOT/'food9_workspace')
    parser.add_argument('--prepared',action='store_true')
    parser.add_argument('--run',type=Path,default=ROOT/'model_runs/food9/source-binary-v1')
    parser.add_argument('--registry',type=Path,default=ROOT/'models/food9.json')
    parser.add_argument('--spoiled-weight',type=float,default=1.)
    parser.add_argument('--minimum-spoiled-recall',type=float,default=0.)
    parser.add_argument('--detail-views',action='store_true',help='Two views at 320 pixels; recipe is stored in the checkpoint.')
    parser.add_argument('--learning-rate',type=float,default=.003)
    args=parser.parse_args()
    base=args.base
    rows=json.loads((base/'split.json').read_text(encoding='utf8')) if args.prepared else prepare(base)
    from foodquality8.source_dataset import validate_rows,verify_files
    validate_rows(rows)
    verify_files(rows)
    print('Ảnh huấn luyện/đánh giá:',len(rows),Counter(r['split'] for r in rows),flush=True)
    from foodquality8.robust_views import DETAIL,LEGACY,extract_features,feature_signature
    recipe=DETAIL if args.detail_views else LEGACY
    model,transform=encoder()
    cache=base/'features.npz'
    signature=feature_signature(rows,recipe)
    signatures={signature}
    if not args.detail_views:
        signatures.add(hashlib.sha256(json.dumps([(r['sha256'],r['split']) for r in rows]).encode()).hexdigest())
    if cache.exists() and str(np.load(cache)['signature']) in signatures:
        features=np.load(cache)['features']
    else:
        chunks=[]
        with torch.inference_mode():
            batch_size=8 if args.detail_views else 32
            for start in range(0,len(rows),batch_size):
                images=[];extra=[]
                for row in rows[start:start+batch_size]:
                    with Image.open(row['path']) as source:
                        im=ImageOps.exif_transpose(source).convert('RGB')
                        images.append(im.copy())
                chunks.append(extract_features(model,images,recipe).numpy())
                if start%320==0: print(f'Trích đặc trưng {start}/{len(rows)}',flush=True)
        features=np.concatenate(chunks)
        np.savez_compressed(cache,features=features,signature=signature)
    masks={s:np.array([r['split']==s for r in rows]) for s in ('train','val','test')}
    mean=features[masks['train']].mean(0);std=np.maximum(features[masks['train']].std(0),.05)
    x=torch.from_numpy(np.clip((features-mean)/std,-8,8))
    y=torch.tensor([FOODS.index(r['food']) for r in rows])
    q=torch.tensor([int(r['condition']=='spoiled') for r in rows])
    torch.manual_seed(42)
    food=torch.nn.Linear(x.shape[1],len(FOODS));quality=torch.nn.Linear(x.shape[1],len(FOODS)*2)
    optimizer=torch.optim.AdamW(list(food.parameters())+list(quality.parameters()),lr=args.learning_rate,weight_decay=.01)
    indices={s:torch.from_numpy(np.flatnonzero(masks[s])) for s in masks}
    def metrics(split):
        ids=indices[split]
        with torch.inference_mode():
            fp=food(x[ids]).argmax(1)
            qp=quality(x[ids]).reshape(-1,len(FOODS),2)[torch.arange(len(ids)),fp].argmax(1)
        true=(y[ids]*2+q[ids]).numpy();pred=(fp*2+qp).numpy()
        names=[f'{f}_{c}' for f in FOODS for c in ('fresh','spoiled')]
        from sklearn.metrics import recall_score
        return dict(food_accuracy=accuracy_score(y[ids],fp),joint_accuracy=accuracy_score(true,pred),
            spoiled_recall=recall_score(true,pred,labels=list(range(1,18,2)),average='macro',zero_division=0),
            joint_macro_f1=f1_score(true,pred,labels=list(range(18)),average='macro',zero_division=0),
            per_class=classification_report(true,pred,labels=list(range(18)),target_names=names,output_dict=True,zero_division=0),samples=len(ids))
    best=-1.;stale=0;history=[]
    run=args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    train=indices['train']
    weights=1/torch.bincount(y[train]*2+q[train],minlength=18).float().clamp_min(1).sqrt()
    weights/=weights.mean()
    for epoch in range(180):
        food.train();quality.train()
        shuffled=train[torch.randperm(len(train))]
        for ids in shuffled.split(256):
            optimizer.zero_grad(set_to_none=True)
            logits=quality(x[ids]).reshape(-1,len(FOODS),2)[torch.arange(len(ids)),y[ids]]
            losses=torch.nn.functional.cross_entropy(food(x[ids]),y[ids],reduction='none',label_smoothing=.05)+torch.nn.functional.cross_entropy(logits,q[ids],reduction='none',label_smoothing=.05)
            loss=(losses*weights[y[ids]*2+q[ids]]*torch.where(q[ids]==1,args.spoiled_weight,1.)).mean()
            loss.backward();optimizer.step()
        food.eval();quality.eval()
        val=metrics('val')
        history.append(dict(epoch=epoch+1,**{k:v for k,v in val.items() if k!='per_class'}))
        if val['joint_macro_f1']>best+1e-5 and val['spoiled_recall']>=args.minimum_spoiled_recall:
            best=val['joint_macro_f1'];stale=0
            torch.save(dict(encoder=model.state_dict(),food_head=food.state_dict(),quality_head=quality.state_dict(),
                mean=torch.from_numpy(mean),std=torch.from_numpy(std),foods=FOODS,
                preprocessing=recipe,recipe='efficientnet_features_plus_hsv_two_heads_v1'),run/'best.pt')
            write_json(run/'validation.json',val)
        else: stale+=1
        if epoch%5==0: print('Vòng',epoch+1,'F1 kiểm tra',round(val['joint_macro_f1'],4),'Recall hỏng',round(val['spoiled_recall'],4),flush=True)
        if stale>=20 and best>=0: break
    write_json(run/'history.json',history)
    if best<0:raise ValueError('No checkpoint met the validation recall requirement.')
    saved=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    food.load_state_dict(saved['food_head']);quality.load_state_dict(saved['quality_head'])
    test=metrics('test')
    write_json(run/'test-metrics.json',test);write_json(run/'history.json',history)
    write_json(run/'data-report.json',json.loads((base/'data-report.json').read_text()))
    registry=dict(weights=str((run/'best.pt').resolve()),sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
        foods=FOODS,conditions=['fresh','spoiled'],source=' + '.join(sorted({r.get('source','FoodClass V2') for r in rows})),
        mode='source_labeled_demo',run=str(run),test={k:v for k,v in test.items() if k!='per_class'},
        note='Không phải kết luận an toàn ăn uống; nhãn nguồn và vùng thu thập chưa được kiểm chứng độc lập.')
    write_json(args.registry,registry)
    print('ĐÃ HOÀN THÀNH FOOD9:',registry['test'],flush=True)

if __name__=='__main__': main()

"""Local imports and explicitly provisional bounding boxes for the simple workflow."""
import csv
import hashlib
import io
import json
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from .model_data import write_json


def normalized(value):
    value=''.join(c for c in unicodedata.normalize('NFD',str(value or '').lower().replace('đ','d')) if unicodedata.category(c)!='Mn')
    return value.strip().replace(' ','_').replace('-','_')


def food_id(value):
    from .google_review import FOOD_ENGLISH
    value=normalized(value)
    return FOOD_ENGLISH.get(value.replace('_',' '),value).replace(' ','_')


def stage_name(value):
    value=normalized(value)
    from .google_review import FOOD_ENGLISH
    for food in sorted(set(FOOD_ENGLISH.values()),key=len,reverse=True):
        if value.startswith(food.replace(' ','_')+'_'):
            value=value[len(food)+1:]
            break
    for stage, aliases in {
        'near_spoiled':('near_spoiled','near_spoilage','sap_hong'),
        'spoiled':('spoiled','rotten','hong'),
        'fresh':('fresh','tuoi'),
    }.items():
        if value in aliases:
            return stage
    return None


def import_local(project, source, food):
    """Copy originals; never move or alter the user's files. Labels come from supplied metadata."""
    source=Path(source).resolve()
    if not source.exists():
        raise ValueError('Không tìm thấy nguồn ảnh đã chọn.')
    identity=hashlib.sha256((str(source)+'|'+food).encode()).hexdigest()[:16]
    folder=Path(project)/'food15_workspace/local_imports'/identity
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    def add(raw, stage, origin):
        digest=hashlib.sha256(raw).hexdigest()
        try:
            with Image.open(io.BytesIO(raw)) as im:
                width,height=im.size
                im.verify()
        except (OSError,ValueError):
            return
        path=folder/(digest+'.img')
        if not path.exists():
            path.write_bytes(raw)
        rows.append(dict(food_code=identity+'_'+digest[:24],img_path=str(path),sha256=digest,
            width=width,height=height,search_stage=stage or 'unlabeled',source_url='',original_source=str(origin),
            query='Nhãn từ dữ liệu người dùng',label_origin='local_metadata'))
    if source.is_dir():
        from .google_review import FOOD_ENGLISH
        known=set(FOOD_ENGLISH.values())
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp'):
                path_foods={food_id(part) for part in path.relative_to(source).parts[:-1]} & known
                if path_foods and food_id(food) not in path_foods:
                    continue
                stage=next((stage_name(part) for part in reversed(path.relative_to(source).parts[:-1]) if stage_name(part)),None)
                add(path.read_bytes(),stage,path)
    elif source.suffix.lower()=='.xlsx':
        from openpyxl import load_workbook
        wb=load_workbook(source)
        try:
            for ws in wb:
                # Toolkit files: A code, B food, C embedded image, D quality, E status, I original path, N query label.
                headers=[str(c.value or '').lower() for c in ws[1]]
                toolkit=len(headers)>=9 and ('ma thuc pham' in headers[0] or 'mã thực phẩm' in headers[0])
                def col(aliases,default=None):
                    return next((i for i,h in enumerate(headers) if h in aliases),default)
                grade_col=3 if toolkit else col(['quality','condition','chat luong','chất lượng'])
                food_col=1 if toolkit else col(['food','food_name','ten thuc pham','tên thực phẩm'])
                path_col=8 if toolkit else col(['path','image_path','đường dẫn','duong dan'])
                pictures={pic.anchor._from.row+1:pic for pic in ws._images if hasattr(pic.anchor,'_from')}
                for n,cells in enumerate(ws.iter_rows(min_row=2,values_only=True),2):
                    if food_col is not None and food_col<len(cells) and cells[food_col] and food_id(cells[food_col])!=food_id(food):
                        continue
                    if toolkit and len(cells)>4 and cells[4]=='Loai bo':
                        continue
                    grade=cells[grade_col] if grade_col is not None and grade_col<len(cells) else None
                    stage=stage_name(grade)
                    if not stage and toolkit and len(cells)>13:
                        stage=stage_name(cells[13])
                    path_value=cells[path_col] if path_col is not None and path_col<len(cells) else None
                    path=Path(str(path_value)) if path_value else None
                    if path and not path.is_absolute():
                        path=source.parent/path
                    if path and path.is_file():
                        add(path.read_bytes(),stage,path)
                    elif n in pictures:
                        add(pictures[n]._data(),stage,f'{source}#{ws.title}!{n}')
        finally:
            wb.close()
    else:
        raise ValueError('Chọn thư mục ảnh hoặc file Excel .xlsx.')
    if not rows:
        raise ValueError('Nguồn chưa có ảnh đọc được. Excel cần ảnh nhúng hoặc cột đường dẫn ảnh.')
    (folder/'manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf8')
    return folder


def propose_box(image):
    """Foreground segmentation heuristic, not a verified food detector."""
    import cv2
    im=ImageOps.exif_transpose(image).convert('RGB')
    im.thumbnail((320,320))
    arr=np.asarray(im)
    if float(arr.std())<5:
        return None
    cv2.setNumThreads(1)
    h,w=arr.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    cv2.setRNGSeed(42)
    cv2.grabCut(arr,mask,(max(1,w//30),max(1,h//30),w-2*max(1,w//30),h-2*max(1,h//30)),
                np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_RECT)
    foreground=np.uint8((mask==1)|(mask==3))
    count,labels,stats,_=cv2.connectedComponentsWithStats(foreground)
    if count<2:
        return None
    best=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    x,y,bw,bh,area=stats[best]
    if area/(w*h)<.035 or bw*bh/(w*h)>.94 or min(bw/w,bh/h)<.1:
        return None
    return dict(x=float((x+bw/2)/w),y=float((y+bh/2)/h),w=float(bw/w),h=float(bh/h))


def add_auto_boxes(release, stop=None, log=print):
    """Build a new release; leave the original classification release immutable."""
    import cv2
    import yaml
    release=Path(release)
    dest=release.with_name(release.name+'-two-models')
    if dest.exists():
        raise ValueError('Bộ dữ liệu tự động đã tồn tại. Hãy chuẩn bị lượt mới.')
    with (release/'metadata.csv').open(encoding='utf8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    accepted=defaultdict(list)
    rejected=[]
    for i,row in enumerate(rows):
        if stop and stop.exists():
            raise InterruptedError('Đã ngừng chuẩn bị dữ liệu.')
        label=row['food_name']+'_'+row['condition']
        path=release/'classification'/row['split']/label/(row['image_id']+'.jpg')
        try:
            with Image.open(path) as image:
                box=propose_box(image)
        except (cv2.error,OSError,ValueError):
            box=None
        if box:
            box['food_name']=row['food_name']
            row.update(bbox_status='Suggested',bbox_json=json.dumps([box]),bbox_origin='grabcut_foreground_suggestion')
            accepted[label].append((row,path,box))
        else:
            rejected.append(row['image_id'])
        log(f'Tự đề xuất vùng: {i+1}/{len(rows)}; loại {len(rejected)} ảnh chưa tách được vùng.')
    expected=json.loads((release/'manifest.json').read_text(encoding='utf8'))['classes']
    write_json(release/'auto-box-report.json',dict(accepted={k:len(v) for k,v in accepted.items()},rejected=rejected,
        method='GrabCut foreground; unverified, may include background or wrong objects'))
    missing=[f'{label}: {len(accepted[label])}/3' for label in expected if len(accepted[label])<3]
    if missing:
        raise ValueError('Thiếu ảnh tách được vùng sau lọc: '+', '.join(missing)+'. Bổ sung ảnh một thực phẩm trên nền đơn giản. Báo cáo: '+str(release/'auto-box-report.json'))
    dest.mkdir()
    foods=sorted({r['food_name'] for r in rows})
    metadata=[]
    for label,items in accepted.items():
        items.sort(key=lambda item:item[0]['sha256'])
        n=max(1,round(len(items)*.15))
        for i,(row,path,box) in enumerate(items):
            row['split']='test' if i<n else 'val' if i<2*n else 'train'
            metadata.append(row)
            split=row['split']
            target=dest/'classification'/split/label/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(path) as original:
                im=ImageOps.exif_transpose(original).convert('RGB')
                left=max(0,int((box['x']-box['w']/2)*im.width))
                top=max(0,int((box['y']-box['h']/2)*im.height))
                right=min(im.width,int((box['x']+box['w']/2)*im.width))
                bottom=min(im.height,int((box['y']+box['h']/2)*im.height))
                im.crop((left,top,right,bottom)).save(target,quality=95)
                row['classification_crop_json']=json.dumps([left,top,right,bottom])
            target=dest/'detection/images'/split/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            annotation=dest/'detection/labels'/split/(path.stem+'.txt')
            annotation.parent.mkdir(parents=True,exist_ok=True)
            annotation.write_text(f"{foods.index(row['food_name'])} {box['x']} {box['y']} {box['w']} {box['h']}\n",encoding='utf8')
    with (dest/'metadata.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    manifest=json.loads((release/'manifest.json').read_text(encoding='utf8'))
    manifest.update(label_mode='automatic_suggested',bbox_mode='grabcut_suggested',
        catalog_sha256=hashlib.sha256((dest/'metadata.csv').read_bytes()).hexdigest())
    write_json(dest/'manifest.json',manifest)
    shutil.copy2(release/'sources.json',dest/'sources.json')
    shutil.copy2(release/'auto-box-report.json',dest/'auto-box-report.json')
    (dest/'detection/data.yaml').write_text(yaml.safe_dump(dict(path=str(dest/'detection'),train='images/train',val='images/val',test='images/test',names=foods)),encoding='utf8')
    return dest

"""Validated edits for the local annotation UI."""
import hashlib
import json
from datetime import datetime, timezone


def box_from_drag(start, end, width, height, food):
    if width <= 0 or height <= 0:
        raise ValueError('Anh chua san sang.')
    left,right=sorted([max(0,min(width,start[0])),max(0,min(width,end[0]))])
    top,bottom=sorted([max(0,min(height,start[1])),max(0,min(height,end[1]))])
    if right-left < 4 or bottom-top < 4:
        raise ValueError('Vung chon qua nho. Hay keo bao quanh thuc pham.')
    x,y=(left+right)/(2*width),(top+bottom)/(2*height)
    w,h=(right-left)/width,(bottom-top)/height
    w,h=min(w,2*min(x,1-x)),min(h,2*min(y,1-y))
    return dict(food_name=food,x=x,y=y,w=w,h=h)


def save_boxes(dataset, image_id, boxes, reviewer, status='Approved'):
    if status not in ('Approved','Pending'):
        raise ValueError('Trang thai bounding box khong hop le.')
    row=next((r for r in dataset.rows() if r['image_id']==image_id),None)
    if row is None:
        raise ValueError('Khong tim thay anh.')
    dataset.validate_boxes(boxes)
    if status=='Approved':
        if not boxes or not reviewer.strip():
            raise ValueError('Can vung khoanh va ten nguoi duyet.')
        if row['error']:
            raise ValueError('Khong the duyet anh loi.')
        if hashlib.sha256((dataset.root/row['raw_path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Anh goc da thay doi.')
        if len(boxes)!=1 or boxes[0]['food_name']!=row['food_name']:
            raise ValueError('Ban dau dung 1 thuc pham/anh. Luu ten thuc pham trong man hinh duyet truoc.')
    with dataset.db:
        dataset.db.execute('UPDATE images SET bbox_json=?,bbox_status=? WHERE image_id=?',
                           (json.dumps(boxes),status,image_id))
        dataset.db.execute("UPDATE images SET split=''")
        dataset.audit('bbox_review',dict(image_id=image_id,boxes=boxes,reviewer=reviewer.strip(),status=status))
    dataset.snapshot()
    dataset.coverage()


def save_source_review(dataset, source_id, status, license_id='', license_url='', source_page='', reviewer='', evidence=''):
    source=dataset.sources()[source_id]
    if status not in ('Approved','Pending','Rejected'):
        raise ValueError('Trang thai nguon khong hop le.')
    if status=='Approved':
        if not all(str(v).strip() for v in (license_id,license_url,source_page,reviewer,evidence)):
            raise ValueError('Can giay phep, URL giay phep, trang nguon, nguoi kiem tra va can cu.')
        if not license_url.startswith(('https://','http://')) or not source_page.startswith(('https://','http://')):
            raise ValueError('Nhap URL trang nguon va giay phep day du.')
    source.update(license_status=status,license_id=license_id.strip(),license_url=license_url.strip(),
                  source_page=source_page.strip(),license_reviewer=reviewer.strip(),license_evidence=evidence.strip(),
                  checked_at=datetime.now(timezone.utc).isoformat(),redistribution_allowed=False)
    with dataset.db:
        dataset.db.execute('UPDATE sources SET config=? WHERE id=?',(json.dumps(source,ensure_ascii=False),source_id))
        dataset.db.execute("UPDATE images SET split=''")
        dataset.audit('source_review',source)
    dataset.snapshot()
    dataset.coverage()

"""GUI-friendly export using existing review, source and group gates."""
from datetime import datetime
from pathlib import Path
from foodquality8 import Dataset


def main():
    root=Path(__file__).resolve().parents[1]/'food15_workspace'
    dataset=Dataset(root)
    try:
        eligible=[r for r in dataset.rows() if dataset.eligible(r)]
        if not eligible:
            raise ValueError('Chua co anh duyet nhan VA duyet nguon. Hoan tat kiem duyet truoc khi xuat.')
        print('Chia train/val/test theo nhom mau...',flush=True)
        dataset.split(seed=42)
        dataset.coverage()
        release=dataset.export('v'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        print('Dataset release:',release,flush=True)
        print('Chon thu muc nay trong hai tab model. YOLO can bounding box da duyet.',flush=True)
    finally:
        dataset.close()


if __name__=='__main__':
    try:
        main()
    except ValueError as exc:
        print('Chua xuat duoc:',exc,flush=True)
        raise SystemExit(1)

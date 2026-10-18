"""Prepare matching frozen features with content-addressed reuse and restart support."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from foodquality8.food9 import encoder
from foodquality8.robust_views import LEGACY,DETAIL
from foodquality8.feature_cache import ensure_features
from foodquality8.source_dataset import validate_rows,verify_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--detail-views',action='store_true');p.add_argument('--reuse',type=Path,action='append',default=[])
    args=p.parse_args();rows=json.loads((args.base/'split.json').read_text(encoding='utf8'))
    validate_rows(rows);verify_files(rows);model,_=encoder()
    values=ensure_features(args.base,rows,model,DETAIL if args.detail_views else LEGACY,args.reuse,
                           log=lambda message:print(message,flush=True))
    print('Đã lưu đặc trưng:',values.shape,flush=True)


if __name__=='__main__':main()

"""One-button dataset augmentation, detail-view training, validation and publication."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from foodquality8.model_data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'workspace/dataset-current.json')
    args=p.parse_args()
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'classifier/best.pt').exists():raise ValueError('Choose a new run directory; checkpoints are immutable.')
    source=ROOT/'workspace/tomato_v3'
    if not (source/'split.json').exists():source=ROOT/'workspace/sic_merged'
    base=ROOT/'workspace/robust9_v1'
    dataset={}
    if args.dataset.exists():
        dataset=json.loads(args.dataset.read_text(encoding='utf8'))
        source=Path(dataset['source_base']);base=Path(dataset['training_base'])
    reuse_training=[Path(v) for v in dataset.get('reuse_training',[])]
    reuse_source=[Path(v) for v in dataset.get('reuse_source',[])]
    raw_quality=run/'quality-detail.json';quality=run/'quality.json';consensus=run/'consensus.json'
    comparison=run/'comparison.json';evaluation=run/'pipeline-validation.json'
    def state(**kw):write_json(run/'workflow.json',kw)
    def execute(script,*arguments):
        process=subprocess.Popen([sys.executable,'-X','utf8','-u',str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT)
        while process.poll() is None:
            if (run/'STOP').exists():
                process.terminate();process.wait()
                state(status='stopped',activated_new_models=False,message='Đã ngừng; giữ mô hình website.');return False
            time.sleep(.2)
        if process.returncode:
            state(status='failed',activated_new_models=False,message='Bước xử lý thất bại; giữ mô hình website, xem nhật ký.');return False
        return True
    state(status='running',message='Tạo biến thể ảnh, học hai vùng 320×320 và kiểm tra tự động.')
    augmentation_reuse=[item for path in reuse_training for item in ('--reuse',path)]
    feature_reuse=[item for path in reuse_training for item in ('--reuse-features',path)]
    legacy_reuse=[item for path in reuse_source for item in ('--reuse',path)]
    commands=[('build_robust_dataset.py','--source',source/'split.json','--output',base,*augmentation_reuse),
              ('cache_source_features.py','--base',source,*legacy_reuse),
              ('train_food9.py','--base',base,'--prepared','--detail-views','--run',run/'classifier','--registry',raw_quality,*feature_reuse),
              ('fit_quality_ensemble.py','--candidate',raw_quality,'--base',base,'--original-base',source,
               '--output',quality,'--checkpoint',run/'classifier/ensemble.pt')]
    for command in commands:
        if not execute(*command):return
    if not quality.exists():
        state(status='complete',activated_new_models=False,message='Đã học xong; chưa có bản cải thiện mà giữ được recall ảnh hỏng, giữ mô hình website.');return
    if not execute('compare_robust_model.py','--candidate',quality,'--base',base,'--original-base',source,'--output',comparison):return
    if not json.loads(comparison.read_text(encoding='utf8'))['passed']:
        state(status='complete',activated_new_models=False,message='Đã học xong; bản mới chưa vượt kiểm tra phân loại, giữ mô hình website.');return
    commands=[('calibrate_consensus.py','--base',base,'--legacy-base',source,'--quality-registry',quality,'--output',consensus),
              ('evaluate_robust_consensus.py','--candidate',consensus,'--manifest',source/'split.json','--output',evaluation,'--full-clean')]
    for command in commands:
        if not execute(*command):return
    if not json.loads(evaluation.read_text(encoding='utf8'))['passed']:
        state(status='complete',activated_new_models=False,message='Đã học xong; luồng ba mô hình chưa vượt kiểm tra, giữ mô hình website.');return
    if not execute('publish_robust.py','--candidate',consensus,'--comparison',comparison,'--evaluation',evaluation):return
    state(status='complete',activated_new_models=True,message='Đã kiểm tra và cập nhật website sang mô hình mới; tải lại trang để sử dụng.')


if __name__=='__main__':main()

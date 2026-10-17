"""One-command source-data retraining, retaining the pretrained SIC detector."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from foodquality8.model_data import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'classifier/best.pt').exists():
        raise ValueError('Thư mục run đã chứa checkpoint. Chọn thư mục mới để không ghi đè bản đã phát hành.')
    def state(**kw):write_json(run/'workflow.json',kw)
    state(status='running',message='Đang học từ dataset kết hợp; giữ YOLO SIC đã huấn luyện.')
    command=[sys.executable,'-X','utf8','-u',str(ROOT/'scripts/train_food9.py'),'--base',str(ROOT/'workspace/sic_merged'),
        '--prepared','--run',str(run/'classifier'),'--registry',str(ROOT/'models/sic/candidate.json')]
    for cmd in (command,[sys.executable,'-X','utf8',str(ROOT/'scripts/compare_sic_models.py')],
                [sys.executable,'-X','utf8','-u',str(ROOT/'scripts/evaluate_fusion.py')],
                [sys.executable,'-X','utf8',str(ROOT/'scripts/publish_sic.py')]):
        process=subprocess.Popen(cmd,cwd=ROOT)
        while process.poll() is None:
            if (run/'STOP').exists():
                process.terminate();process.wait()
                state(status='stopped',message='Đã ngừng; mô hình đang phục vụ website không thay đổi.');return
            time.sleep(.2)
        if process.returncode:
            state(status='failed',message='Huấn luyện/kiểm tra thất bại; xem nhật ký.');return
    state(status='complete',activated_new_models=True,
        message='Đã học, so sánh và kiểm tra luồng kết hợp. Website đã cập nhật; tải lại trang để dùng bản mới. Kết quả: workspace/sic_merged.')
    print('Hoàn tất huấn luyện, kiểm tra và cập nhật website.',flush=True)

if __name__=='__main__':main()

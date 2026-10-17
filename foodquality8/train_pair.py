"""One command: prepare, propose regions, train both models, publish a complete pair."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .model_data import write_json


def run_pair(project, source, food, run, epochs=20, batch=4, excel=None):
    from .auto_workflow import import_local, add_auto_boxes
    from .quick_dataset import prepare_google, STAGES
    from .inference import activate_run
    project,run=Path(project).resolve(),Path(run).resolve()
    run.mkdir(parents=True,exist_ok=True)
    stop=run/'STOP'
    state=dict(status='preparing',food=food,source=str(source))
    write_json(run/'workflow.json',state)
    try:
        if source=='google':
            release=prepare_google(project,food,excel)
        else:
            folder=import_local(project,source,food)
            rows=[json.loads(line) for line in (folder/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
            edits=folder/'corrections.json'
            edits=json.loads(edits.read_text(encoding='utf8')) if edits.exists() else {}
            stages=sorted({edits.get(r['food_code'],{}).get('stage',r['search_stage']) for r in rows}&set(STAGES))
            if len(stages)<2:
                raise ValueError('Ảnh có sẵn cần ít nhất 2 mức chất lượng. Dùng thư mục con Tuoi / Sap hong / Hong, Excel có chất lượng, hoặc Sửa nhãn tùy chọn. Không thể suy ra độ hỏng chỉ từ tên thực phẩm.')
            release=prepare_google(project,food,source_folder=folder,required_stages=stages)
        if stop.exists():
            raise InterruptedError('Đã ngừng trước khi huấn luyện.')
        release=add_auto_boxes(release,stop)
        state['release']=str(release)
        for kind,title in [('classifier','1/2 · Học chất lượng'),('detector','2/2 · Học khoanh vùng')]:
            if stop.exists():
                raise InterruptedError('Đã ngừng. Không chạy mô hình tiếp theo.')
            state['status']=kind
            write_json(run/'workflow.json',state)
            print(title,flush=True)
            child_run=run/kind
            args=[sys.executable,'-u','-m','foodquality8.training',kind,'--release',str(release),
                '--run',str(child_run),'--epochs',str(epochs),'--batch',str(batch),'--experimental']
            child=subprocess.Popen(args,cwd=project,env={**os.environ,'PYTHONUTF8':'1'})
            try:
                while child.poll() is None:
                    if stop.exists() and child_run.exists():
                        (child_run/'STOP').touch()
                    time.sleep(.2)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
            if stop.exists():
                raise InterruptedError('Đã ngừng. Kết quả đã học được giữ lại; chưa thay mô hình website.')
            if child.returncode:
                raise ValueError(f'{title} chưa hoàn thành. Xem lỗi phía trên; chưa thay mô hình website.')
            info=json.loads((child_run/'run.json').read_text(encoding='utf8'))
            if info.get('status')!='complete':
                raise ValueError('Mô hình chưa hoàn tất đánh giá; không kích hoạt cặp mô hình.')
        candidate=run/'candidate-bundle.json'
        activate_run(run/'classifier',candidate,allow_experimental=True)
        activate_run(run/'detector',candidate,allow_experimental=True)
        if stop.exists():
            raise InterruptedError('Đã ngừng trước khi cập nhật website.')
        active=project/'models/active.json'
        active.parent.mkdir(parents=True,exist_ok=True)
        if active.exists():
            shutil.copy2(active,run/'previous-bundle.json')
        from .model_selection import choose_bundle
        previous=json.loads(active.read_text(encoding='utf8')) if active.exists() else {}
        chosen,decisions=choose_bundle(json.loads(candidate.read_text(encoding='utf8')),previous)
        write_json(run/'selection.json',decisions)
        write_json(active,chosen)
        state['selection']=decisions
        state['activated_new_models']=[k for k,v in chosen.items() if v!=previous.get(k)]
        state['status']='complete'
        (run/'report.html').write_text('''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Kết quả huấn luyện hai mô hình</title><style>body{font:18px system-ui;max-width:850px;margin:50px auto;color:#164c3d}a{display:block;padding:20px;background:#edf7f0;margin:15px 0}</style>
<h1>Đã hoàn tất hai mô hình</h1><p>Nhãn chất lượng và vùng tự động chưa được kiểm chứng. Điểm kiểm tra là mức khớp với nhãn đề xuất, chưa phải độ chính xác thực tế.</p>
<p>Website chỉ nhận mô hình mới nếu tốt hơn trên cùng tập kiểm tra. Nếu không, giữ mô hình trước đó. Xem quyết định trong selection.json.</p>
<a href="classifier/report.html">1. Kết quả học chất lượng</a><a href="detector/report.html">2. Kết quả học khoanh vùng</a>
<a href="http://127.0.0.1:18501">Mở website tra cứu</a></html>''',encoding='utf8')
        print('HOÀN THÀNH CẢ HAI MÔ HÌNH. Kết quả chọn mô hình: '+json.dumps(decisions,ensure_ascii=False),flush=True)
    except InterruptedError as exc:
        state.update(status='stopped',message=str(exc))
        print(str(exc),flush=True)
    except Exception as exc:
        state.update(status='failed',message=str(exc))
        raise
    finally:
        write_json(run/'workflow.json',state)
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--food',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--excel')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.epochs<=500 or not 1<=args.batch<=128:
        parser.error('Số vòng học 1–500; ảnh/lượt 1–128.')
    try:
        run_pair(**vars(args))
    except Exception as exc:
        parser.exit(1,f'Chưa hoàn thành: {exc}\n')


if __name__=='__main__':
    main()

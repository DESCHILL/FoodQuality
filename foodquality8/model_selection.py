"""Compare validation only, never use test scores to select the deployed weights."""
import json
from pathlib import Path


def validation_improved(candidate, current, kind):
    candidate,current=Path(candidate),Path(current)
    new=json.loads((candidate/'run.json').read_text(encoding='utf8'))
    old=json.loads((current/'run.json').read_text(encoding='utf8'))
    if new['dataset'].get('dataset_sha256')!=old['dataset'].get('dataset_sha256'):
        return False,'Bộ dữ liệu khác nhau; chưa có tập kiểm tra chung độc lập để so sánh.'
    if not (candidate/'validation.json').exists() or not (current/'validation.json').exists():
        return False,'Thiếu kết quả trên tập kiểm tra để so sánh.'
    a=json.loads((candidate/'validation.json').read_text(encoding='utf8'))
    b=json.loads((current/'validation.json').read_text(encoding='utf8'))
    if kind=='classifier':
        # Do not improve aggregate F1 by sacrificing recall for spoiled food.
        for label,metrics in b.get('per_class',{}).items():
            if label.endswith('_spoiled') and not label.endswith('_near_spoiled'):
                if a.get('per_class',{}).get(label,{}).get('recall',0)<metrics.get('recall',0):
                    return False,'Độ bao phủ lớp hỏng giảm; giữ mô hình cũ.'
        passed=a['macro_f1']>b['macro_f1']+1e-8
    else:
        passed=a['map50_95']>b['map50_95']+1e-4 and a['recall']>=b['recall']-.01
    return passed,'Tốt hơn trên tập kiểm tra cùng dữ liệu.' if passed else 'Chưa tốt hơn mô hình cũ trên tập kiểm tra.'


def choose_bundle(candidate,active):
    if not active:
        return candidate,{k:'Chưa có mô hình trước đó; dùng bản thử nghiệm đầu tiên.' for k in candidate}
    if set(candidate)=={'classifier','detector'}:
        foods={r['food'] for r in candidate['classifier'].get('class_labels',[])}
        old_foods={r['food'] for r in active.get('classifier',{}).get('class_labels',[])}
        if foods!=old_foods and set(candidate['detector'].get('classes',[]))<=foods:
            return candidate,{'pair':'Đổi phạm vi thực phẩm theo lượt train mới; không so sánh độ chính xác giữa các loại khác nhau.'}
    chosen=dict(active)
    decisions={}
    for kind,entry in candidate.items():
        if kind not in active:
            chosen[kind]=entry
            decisions[kind]='Chưa có mô hình trước đó.'
            continue
        passed,reason=validation_improved(entry['run'],active[kind]['run'],kind)
        decisions[kind]=reason
        if passed:
            chosen[kind]=entry
    if set(chosen.get('detector',{}).get('classes',[])) - {r['food'] for r in chosen.get('classifier',{}).get('class_labels',[])}:
        return active,{'pair':'Không thay cặp mô hình vì phạm vi thực phẩm không khớp.'}
    return chosen,decisions

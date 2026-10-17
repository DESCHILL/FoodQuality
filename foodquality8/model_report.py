import html
import json
from pathlib import Path


def make_report(run):
    run=Path(run)
    config=json.loads((run/'run.json').read_text(encoding='utf8'))
    metrics=json.loads((run/'test-metrics.json').read_text(encoding='utf8'))
    keys=('accuracy','macro_f1','loss','samples') if config['kind']=='classifier' else ('map50','map50_95','precision','recall')
    titles=dict(accuracy='Tỷ lệ khớp nhãn',macro_f1='F1 trung bình',loss='Sai số',samples='Số ảnh',
        map50='Điểm khoanh vùng mAP50',map50_95='Điểm khoanh vùng mAP50–95',precision='Độ chuẩn xác',recall='Độ bao phủ')
    rows=''.join(f'<tr><th>{titles[key]}</th><td>{metrics[key]:.4f}</td></tr>' for key in keys)
    details=''
    if config['kind']=='classifier':
        details='<h2>Kết quả từng lớp trên tập đánh giá</h2><table><tr><th>Lớp</th><th>Độ chuẩn xác</th><th>Độ bao phủ</th><th>F1</th><th>Số ảnh</th></tr>'
        for name in metrics['classes']:
            r=metrics['per_class'][name]
            details+=f'<tr><td>{html.escape(name)}</td><td>{r["precision"]:.4f}</td><td>{r["recall"]:.4f}</td><td>{r["f1-score"]:.4f}</td><td>{r["support"]}</td></tr>'
        details+='</table><h2>Bảng nhầm lẫn giữa các lớp</h2><p>Hàng: nhãn trong dữ liệu. Cột: nhãn dự đoán. Thứ tự lớp như bảng trên.</p><table>'
        for row in metrics['confusion_matrix']:
            details+='<tr>'+''.join(f'<td>{n}</td>' for n in row)+'</tr>'
        details+='</table>'
    warning='<p><strong>THỬ NGHIỆM VỚI NHÃN/VÙNG TỰ ĐỘNG: số đo dưới đây chỉ là mức khớp nhãn gợi ý, không phải độ chính xác thực phẩm đã kiểm chứng. Quyền sử dụng nguồn chưa được xác minh.</strong></p>' if config['dataset'].get('label_mode') in ('google_query_suggested','automatic_suggested') else ''
    document=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kết quả FoodQuality</title>
<style>body{{font:16px system-ui;margin:40px auto;max-width:1050px;color:#193d30;background:#f6faf7}}table{{border-collapse:collapse;background:white}}th,td{{padding:10px 16px;border:1px solid #d3e3d8;text-align:left}}h1{{color:#175d42}}</style>
<h1>FoodQuality · {html.escape(config['kind'])}</h1>{warning}<p>Thiết bị: {html.escape(config['device'])} · Số vòng tối đa: {config['epochs']}</p>
<p>Dataset: {html.escape(config['dataset']['root'])}</p><h2>Đánh giá trên tập test</h2><table>{rows}</table>{details}
<p>Số liệu phản ánh bộ test của lượt chạy này; không bảo đảm độ chính xác ngoài thực tế. Chất lượng ngoại quan không phải kết luận an toàn ăn uống.</p></html>'''
    (run/'report.html').write_text(document,encoding='utf8')

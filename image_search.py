"""Google image collection with resumable originals, provenance and Excel previews.

Search terms are NOT freshness labels. All downloaded candidates require review.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import threading
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import requests
from PIL import Image, ImageOps
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


SEARCH_STAGES = [
    ('fresh', 'Tươi', 'tươi', 'fresh'),
    ('spoiled', 'Hỏng', 'hỏng', 'spoiled'),
]
FOOD_ENGLISH = {
    'tao': 'apple', 'chuoi': 'banana', 'cam': 'orange', 'xoai': 'mango',
    'dau tay': 'strawberry', 'nho': 'grape', 'oi': 'guava',
    'ca chua': 'tomato', 'khoai tay': 'potato', 'dua chuot': 'cucumber',
    'dua leo': 'cucumber', 'ca rot': 'carrot', 'ot chuong': 'bell pepper',
    'thit bo': 'beef', 'bo': 'beef', 'thit lon': 'pork', 'thit heo': 'pork',
    'lon': 'pork', 'heo': 'pork', 'thit ga': 'chicken meat', 'ga': 'chicken meat',
    'ca': 'fish', 'du du': 'papaya', 'dua': 'pineapple', 'thom': 'pineapple',
    'ca tim': 'eggplant', 'kho qua': 'bitter melon', 'muop dang': 'bitter melon',
    'thit cuu': 'mutton', 'cuu': 'mutton',
}


def build_search_plan(food_name):
    """Deterministic query templates; no AI or generated freshness labels."""
    name = ' '.join(food_name.strip().split())
    if not name:
        raise ValueError('Nhap ten thuc pham.')
    normalized = ''.join(c for c in unicodedata.normalize('NFD', name.lower().replace('đ', 'd'))
                         if unicodedata.category(c) != 'Mn')
    english = FOOD_ENGLISH.get(normalized, name)
    plan = []
    # Visit all stages before switching language so fresh images cannot fill the target first.
    for language in ('vi', 'en'):
        for key, label, vi, en in SEARCH_STAGES:
            query = f'{name} {vi} thực phẩm' if language == 'vi' else f'{english} {en} food'
            plan.append(dict(stage=key, stage_name=label, query=query, language=language))
    return plan


def food_key(name):
    slug = re.sub(r'[^\w-]+', '_', name.strip().lower(), flags=re.UNICODE)[:60]
    return slug or 'food'


def sheet_name(name):
    safe = re.sub(r'[\\/*?\[\]:]', '_', name).strip("'") or 'Food'
    return safe if len(safe) <= 31 else safe[:24] + '_' + hashlib.sha256(name.encode()).hexdigest()[:6]


def _text(cell, value):
    # Untrusted search terms/URLs must never become spreadsheet formulas.
    cell.value = str(value or '')
    cell.data_type = 's'


def export_excel(food_name, rows, excel_path, log_cb=print):
    """Embed PNG bytes in XLSX; preserve other foods and manually reviewed labels."""
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(excel_path) if excel_path.exists() else Workbook()
    title = sheet_name(food_name)
    reviews = {}
    if title in wb.sheetnames:
        old = wb[title]
        if old.cell(1, 4).value == 'Chat luong thuc pham (duyet)':
            reviews = {r[0]: (r[3], r[4]) for r in old.iter_rows(min_row=2, values_only=True) if r[0]}
        del wb[title]
    if 'Sheet' in wb.sheetnames and wb['Sheet'].max_row == 1 and wb['Sheet']['A1'].value is None:
        del wb['Sheet']
    ws = wb.create_sheet(title)
    headers = ['Ma thuc pham', 'Ten thuc pham', 'Anh thuc pham',
               'Chat luong thuc pham (duyet)', 'Trang thai duyet', 'Tu khoa tim (khong phai nhan)',
               'URL anh goc', 'Trang tim Google', 'Tep anh tren may', 'Do phan giai',
               'Thoi gian tai UTC', 'Quyen su dung', 'SHA256', 'Nhãn gợi ý từ truy vấn', 'Kết quả lọc kỹ thuật']
    widths = [24, 20, 24, 25, 20, 30, 45, 40, 40, 16, 24, 25, 32, 28, 35]
    for col, (header, width) in enumerate(zip(headers, widths), 1):
        cell = ws.cell(1, col, header)
        cell.font = Font(color='FFFFFF', bold=True)
        cell.fill = PatternFill('solid', fgColor='174A45')
        cell.alignment = Alignment(wrap_text=True, vertical='center')
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[1].height = 34
    for idx, row in enumerate(rows, 2):
        label, status = reviews.get(row['food_code'], ('Chua xac dinh', 'Cho duyet'))
        if status in ('Cho duyet','Goi y tu truy van') and label=='Chua xac dinh':
            label={'fresh':'Tuoi','near_spoiled':'Sap hong','spoiled':'Hong'}.get(row.get('search_stage'),'Chua xac dinh')
            if label!='Chua xac dinh':
                status='Goi y tu truy van'
        values = [row['food_code'], food_name, '', label, status, row.get('query', ''),
                  row.get('source_url', ''), row.get('google_url', ''), str(row['img_path']),
                  f"{row['width']}x{row['height']}", row.get('downloaded_at', ''),
                  'Chua xac minh giay phep', row['sha256'], row.get('search_stage_name', 'Chua ghi nhan'), row.get('filter_result','Chưa kiểm tra lại')]
        for col, value in enumerate(values, 1):
            _text(ws.cell(idx, col), value)
            ws.cell(idx, col).alignment = Alignment(vertical='center', wrap_text=True)
        for col in (4, 5):
            ws.cell(idx, col).fill = PatternFill('solid', fgColor='FFF2CC')
        try:
            with Image.open(row['img_path']) as original:
                preview = ImageOps.exif_transpose(original).convert('RGB')
                preview.thumbnail((160, 110))
                buffer = io.BytesIO()
                preview.save(buffer, format='PNG')
                buffer.seek(0)
                picture = XLImage(buffer)
                ws.add_image(picture, f'C{idx}')
        except (OSError,ValueError):
            _text(ws.cell(idx,3),'Không đọc được ảnh — loại khỏi huấn luyện')
        ws.row_dimensions[idx].height = 85
    for col, options in [('D', 'Chua xac dinh,Tuoi,Hong'),
                         ('E', 'Goi y tu truy van,Cho duyet,Da duyet,Loai bo')]:
        validation = DataValidation(type='list', formula1=f'"{options}"')
        ws.add_data_validation(validation)
        validation.add(f'{col}2:{col}{max(2, len(rows)+1)}')
    ws.freeze_panes = 'D2'
    ws.auto_filter.ref = ws.dimensions
    temporary = excel_path.with_name(excel_path.stem + '.writing.xlsx')
    wb.save(temporary)
    wb.close()
    try:
        temporary.replace(excel_path)
        return excel_path
    except PermissionError:
        fallback = excel_path.with_name(excel_path.stem + '-moi.xlsx')
        temporary.replace(fallback)
        log_cb(f'[Excel dang mo] Da luu ban moi: {fallback}')
        return fallback


def image_candidates(content):
    """Original image URLs embedded by Google, never embedded thumbnail data."""
    content = html.unescape(content).replace('\\u003d', '=').replace('\\u0026', '&').replace('\\/', '/')
    found = []
    # Google result data contains [URL, height, width] tuples.
    for url in re.findall(r'\["(https?://[^"\s]+)",\s*\d+,\s*\d+\]', content):
        found.append(url)
    for href in re.findall(r'href="([^"]+)"', content):
        found.extend(parse_qs(urlparse(href).query).get('imgurl', []))
    # Preview panels can expose originals as ordinary image elements.
    found.extend(re.findall(r'<img[^>]+src="(https?://[^"\s]+)"', content))
    result = []
    for url in found:
        host = (urlparse(url).hostname or '').lower()
        if not host or any(x in host for x in ('google.', 'gstatic.', 'googleusercontent.')):
            continue
        if url not in result:
            result.append(url)
    return result


def _download(session, url, folder, stop_event, excluded_hashes=None):
    """Bounded, interruptible download; validate bytes and deduplicate decoded pixels."""
    with session.get(url, stream=True, timeout=(5, 5)) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            if stop_event.is_set():
                return None
            size += len(chunk)
            if size > 25 * 1024 * 1024:
                raise ValueError('Anh lon hon 25 MB')
            chunks.append(chunk)
        data = b''.join(chunks)
    with Image.open(io.BytesIO(data)) as original:
        if original.width * original.height > 40_000_000:
            raise ValueError('Anh qua lon')
        im = ImageOps.exif_transpose(original).convert('RGB')
        if min(im.size) < 200:
            raise ValueError('Anh nho hon 200px')
        pixels = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        if excluded_hashes and pixels in excluded_hashes:
            return None
        digest = hashlib.sha256(data).hexdigest()
        ext = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif', 'BMP': '.bmp'}.get(original.format, '.img')
        path = folder / (digest + ext)
        if not path.exists():
            temp = path.with_suffix('.part')
            temp.write_bytes(data)
            temp.replace(path)
        return dict(img_path=str(path.resolve()), sha256=digest, pixel_hash=pixels,
                    width=im.width, height=im.height, source_url=url,
                    downloaded_at=datetime.now(timezone.utc).isoformat())


def _load_rows(folder, food_name):
    manifest = folder / 'manifest.jsonl'
    rows = []
    if manifest.exists():
        for line in manifest.read_text(encoding='utf8').splitlines():
            try:
                row = json.loads(line)
                if Path(row['img_path']).is_file():
                    rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def search_and_export(food_name, max_num, downloads_root, excel_path, stop_event, log_cb):
    """Run until target total (0=unlimited) or Stop; always export saved candidates."""
    food_name = food_name.strip()
    if not food_name or max_num < 0:
        raise ValueError('Nhap ten thuc pham va so anh >= 0.')
    folder = Path(downloads_root) / food_key(food_name)
    folder.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(folder, food_name)
    from foodquality8.quick_dataset import suggest_stage, filter_image
    counts=Counter()
    for row in rows:
        reason, valid=filter_image(Path(row['img_path']),row.get('source_url',''))
        row['filter_result']=reason
        if valid:
            counts[suggest_stage(row)]+=1
    stage_count=len(SEARCH_STAGES)
    quotas={stage[0]:max_num//stage_count+(i<max_num%stage_count) for i,stage in enumerate(SEARCH_STAGES)}
    def enough():
        return bool(max_num) and (all(counts[s]>=n for s,n in quotas.items()) if max_num>=stage_count else len(rows)>=max_num)
    hashes = {r.get('pixel_hash', r['sha256']) for r in rows}
    seen_urls = {r.get('source_url') for r in rows}
    final_path = Path(excel_path)
    target = str(max_num) if max_num else 'khong gioi han'
    log_cb(f'[Bat dau] {food_name}: {len(rows)}/{target} anh. 0 = chay den khi NGUNG.')
    log_cb('Anh Google la du lieu cho duyet; tu khoa tim KHONG phai nhan tuoi/hong.')
    if stop_event.is_set():
        if rows:
            return str(export_excel(food_name, rows, final_path, log_cb))
        raise RuntimeError('Da ngung, chua co anh de xuat.')
    if enough():
        return str(export_excel(food_name, rows, final_path, log_cb))
    from playwright.sync_api import sync_playwright, TimeoutError as BrowserTimeout, Error as BrowserError
    plan = build_search_plan(food_name)
    log_cb('[Tự tạo lệnh] 2 mức: Tươi / Hỏng, tiếng Việt và tiếng Anh:')
    for item in plan:
        log_cb(f'  {item["stage_name"]}: {item["query"]}')
    (folder / 'search-plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf8')
    batch_size = max(1, min(10, (max_num + stage_count-1) // stage_count)) if max_num else 10
    try:
        with sync_playwright() as driver:
            # Separate profile, no access to the user's personal browser profile.
            context = driver.chromium.launch_persistent_context(
                str(Path(downloads_root).resolve() / '_google_browser'), channel='msedge',
                headless=False, viewport={'width': 1100, 'height': 760})
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(3000)
            session = requests.Session()
            session.headers['User-Agent'] = 'Mozilla/5.0 FoodQualityDatasetCollector/1.0'
            round_number = 0
            try:
                while not stop_event.is_set() and not enough():
                    item = plan[round_number % len(plan)]
                    if max_num>=stage_count and counts[item['stage']]>=quotas[item['stage']]:
                        round_number+=1
                        continue
                    query = item['query']
                    offset = ((round_number // len(plan)) % 10) * 100
                    google_url = 'https://www.google.com/search?' + urlencode({'q': query, 'udm': 2, 'start': offset})
                    log_cb(f'[Google - {item["stage_name"]}] Luot {round_number+1}: {query}, trang {offset//100+1}')
                    try:
                        page.goto(google_url, wait_until='domcontentloaded', timeout=15000)
                    except BrowserTimeout:
                        log_cb('[Cho] Trang Google tai cham, dang kiem tra...')
                    except BrowserError as exc:
                        if page.is_closed():
                            log_cb('[Da dong Edge] Dung va luu anh da tai.')
                            break
                        log_cb(f'[Loi mang Google] {str(exc)[:120]}. Thu lai sau 30 giay.')
                        stop_event.wait(30)
                        continue
                    if stop_event.wait(2):
                        break
                    notified = False
                    while not stop_event.is_set():
                        body = page.locator('body').inner_text().lower()
                        blocked = ('/sorry/' in page.url or 'unusual traffic' in body or
                                   'lưu lượng truy cập bất thường' in body or 'consent.google.' in page.url)
                        if not blocked:
                            break
                        if not notified:
                            log_cb('[Can xac minh] Hay tu xac minh Google trong cua so Edge. Tool se cho; NGUNG van hoat dong.')
                            notified = True
                        stop_event.wait(2)
                    if stop_event.is_set():
                        break
                    before = len(rows)
                    for scroll in range(12):
                        if stop_event.is_set() or enough() or (max_num>=stage_count and counts[item['stage']]>=quotas[item['stage']]) or len(rows) - before >= batch_size:
                            break
                        for url in image_candidates(page.content()):
                            if stop_event.is_set() or enough() or (max_num>=stage_count and counts[item['stage']]>=quotas[item['stage']]) or len(rows) - before >= batch_size:
                                break
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)
                            try:
                                row = _download(session, url, folder, stop_event, hashes)
                                if not row or row['pixel_hash'] in hashes:
                                    continue
                                reason,valid=filter_image(Path(row['img_path']),url)
                                if not valid:
                                    log_cb('[Lọc ảnh] '+reason)
                                    continue
                                row['filter_result']=reason
                                row.update(food_code=food_key(food_name)[:20] + '_' + row['sha256'][:12],
                                           food_name=food_name, query=query, google_url=google_url,
                                           search_stage=item['stage'], search_stage_name=item['stage_name'],
                                           freshness='unknown', review_status='pending', license='unverified')
                                hashes.add(row['pixel_hash'])
                                with (folder / 'manifest.jsonl').open('a', encoding='utf8') as manifest:
                                    manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
                                rows.append(row)
                                counts[item['stage']]+=1
                                log_cb('Theo mức tìm kiếm: '+', '.join(f'{label}: {counts[key]}' for key,label,_,_ in SEARCH_STAGES))
                                log_cb(f'[Anh {len(rows)}/{target}] {row["width"]}x{row["height"]} - {url[:100]}')
                                if len(rows) % 10 == 0:
                                    final_path = export_excel(food_name, rows, excel_path, log_cb)
                            except (requests.RequestException, OSError, ValueError) as exc:
                                log_cb(f'[Bo qua anh] {str(exc)[:140]}')
                        page.evaluate('window.scrollBy(0, window.innerHeight * 2)')
                        if stop_event.wait(1):
                            break
                    round_number += 1
                    if len(rows) == before:
                        log_cb('[Chua co anh moi] Cho 30 giay roi tim tiep. Co the bam NGUNG de luu ket qua.')
                        stop_event.wait(30)
                    else:
                        stop_event.wait(2)
            finally:
                session.close()
                context.close()
    finally:
        if rows:
            final_path = export_excel(food_name, rows, excel_path, log_cb)
            log_cb(f'[Da luu] {len(rows)} anh co hinh nhung trong Excel: {final_path}')
        else:
            log_cb('[Chua co anh] Khong tao Excel rong. Can Google tra ket qua va URL anh tai duoc.')
    if not rows:
        raise RuntimeError('Chua tai duoc anh. Xem nhat ky Google; khong co dataset de xuat.')
    return str(final_path)

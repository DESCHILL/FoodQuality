"""Manual image review UI, backed by the project catalog rather than guessed labels."""
import json
import tkinter as tk
import webbrowser
from collections import Counter
from tkinter import ttk, messagebox

from PIL import Image, ImageOps, ImageTk
from foodquality8 import Dataset
from foodquality8.google_review import import_google, save_review


class ReviewWindow(tk.Toplevel):
    def __init__(self, parent, project):
        super().__init__(parent)
        self.project = project
        self.title('FoodQuality - Duyet anh va gan nhan')
        self.geometry('1050x800')
        self.dataset = Dataset(project / 'food15_workspace')
        if not self.dataset.taxonomy and not self.dataset.rows():
            self.dataset.set_taxonomy(project / 'config/food15-taxonomy.json')
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.position = 0
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='Nhap / cap nhat anh Google', command=self.sync).pack(side='left')
        ttk.Button(toolbar, text='Mo nguon anh', command=self.open_source).pack(side='left', padx=8)
        ttk.Button(toolbar, text='Bao cao do phu', command=self.report).pack(side='left')
        ttk.Button(toolbar, text='Khoanh vung YOLO', command=self.box_editor).pack(side='left',padx=8)
        ttk.Button(toolbar, text='Kiem tra quyen nguon', command=self.source_editor).pack(side='left')
        ttk.Button(toolbar, text='Xuất bộ dữ liệu', command=self.export_dataset).pack(side='left',padx=4)
        self.summary = tk.StringVar()
        ttk.Label(self, textvariable=self.summary, wraplength=1000).pack(fill='x', padx=10)
        self.picture = ttk.Label(self, anchor='center')
        self.picture.pack(fill='both', expand=True, padx=10, pady=5)
        self.info = tk.StringVar()
        ttk.Label(self, textvariable=self.info, wraplength=1000).pack(fill='x', padx=10)
        form = ttk.Frame(self, padding=10)
        form.pack(fill='x')
        self.food = tk.StringVar()
        self.condition = tk.StringVar()
        self.reviewer = tk.StringVar()
        self.evidence = tk.StringVar()
        self.group = tk.StringVar()
        for i, (title, variable) in enumerate([('Thuc pham', self.food), ('Chat luong', self.condition),
                    ('Nguoi duyet', self.reviewer), ('Can cu / thong tin nguon', self.evidence), ('Nhom cung mau / phien chup', self.group)]):
            ttk.Label(form, text=title).grid(row=i, column=0, sticky='w', pady=3)
            if i < 2:
                widget = ttk.Combobox(form, textvariable=variable, state='readonly')
                if i == 0:
                    widget['values'] = ['unknown', *self.dataset.foods]
                    widget.bind('<<ComboboxSelected>>', self.food_changed)
                else:
                    self.condition_box = widget
            else:
                widget = ttk.Entry(form, textvariable=variable)
            widget.grid(row=i, column=1, sticky='ew', padx=8)
        form.columnconfigure(1, weight=1)
        ttk.Label(self, text='fresh = tuoi | semi_fresh = it tuoi | spoiled = hong | unknown = chua xac dinh.\n'
                  'Chi dung cac muc da cau hinh cho tung thuc pham. Tu khoa Google khong phai bang chung nhan.').pack(padx=10)
        buttons = ttk.Frame(self, padding=10)
        buttons.pack(fill='x')
        for label, command in [('Anh truoc', lambda: self.move(-1)), ('Anh sau', lambda: self.move(1)),
                               ('Luu cho duyet', lambda: self.save('Pending')), ('Duyet va tiep', lambda: self.save('Approved')),
                               ('Loai anh va tiep', lambda: self.save('Rejected'))]:
            ttk.Button(buttons, text=label, command=command).pack(side='left', padx=3)
        self.reload()

    def close(self):
        if self.dirty() and not messagebox.askyesno('Chua luu','Bo cac thay doi chua luu va dong?',parent=self):
            return
        self.dataset.close()
        self.destroy()

    def draft(self):
        return tuple(v.get() for v in (self.food,self.condition,self.reviewer,self.evidence,self.group))

    def dirty(self):
        return hasattr(self,'saved_draft') and self.draft()!=self.saved_draft

    def require_saved(self):
        if self.dirty():
            messagebox.showinfo('Luu nhan truoc','Bam Luu cho duyet hoac Duyet va tiep truoc khi mo chuc nang nay.',parent=self)
            return False
        return True

    def reload(self):
        self.rows = self.dataset.rows()
        counts = Counter(r['review_status'] for r in self.rows)
        self.summary.set(f"Anh dataset: {len(self.rows)} | Cho duyet: {counts['Pending']} | Da duyet nhan: {counts['Approved']} | Loai: {counts['Rejected']}")
        self.position = min(self.position, max(0, len(self.rows)-1))
        self.show()

    def show(self):
        if not self.rows:
            self.info.set('Chua nhap anh. Bam Nhap / cap nhat anh Google.')
            return
        row = self.rows[self.position]
        try:
            with Image.open(self.dataset.root / row['raw_path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((850, 350))
                self.photo = ImageTk.PhotoImage(im)
            self.picture.configure(image=self.photo, text='')
        except Exception as exc:
            self.picture.configure(image='', text=f'Loi doc anh: {exc}')
        self.food.set(row['food_name'])
        self.food_changed()
        self.condition.set(row['condition'])
        if row['reviewer']:
            self.reviewer.set(row['reviewer'])
        self.evidence.set(row['label_evidence'])
        self.group.set(row['capture_group'])
        self.saved_draft=self.draft()
        source = self.dataset.sources()[row['source_dataset']]
        self.info.set(f"Anh {self.position+1}/{len(self.rows)} | {row['image_id']} | {row['review_status']} | Ban sao: {row['duplicate_of'] or 'khong'}\n"
                      f"Tu khoa: {source.get('search_query', '')}\nQuyen su dung: {source.get('license_status', 'Pending')} | Box: {row['bbox_status']} | Nhom gan trung: {row['group_id']}")

    def box_editor(self):
        if not self.rows or not self.require_saved():
            return
        row=self.rows[self.position]
        if row['food_name']=='unknown':
            messagebox.showwarning('Chua co ten','Chon thuc pham va bam Luu cho duyet truoc.',parent=self)
            return
        from annotation_gui import BoxWindow
        window=BoxWindow(self,row,self.reviewer.get())
        window.grab_set()

    def export_dataset(self):
        from dataset_export_gui import ExportWindow
        window=ExportWindow(self,self.project)
        window.grab_set()

    def source_editor(self):
        if self.rows and self.require_saved():
            from annotation_gui import SourceWindow
            window=SourceWindow(self,self.rows[self.position])
            window.grab_set()

    def food_changed(self, event=None):
        choices = ['unknown', *self.dataset.states.get(self.food.get(), [])]
        self.condition_box['values'] = choices
        if self.condition.get() not in choices:
            self.condition.set('unknown')

    def move(self, offset):
        if self.rows:
            if self.dirty() and not messagebox.askyesno('Chua luu','Bo thay doi chua luu de chuyen anh?',parent=self):
                return
            self.position = (self.position + offset) % len(self.rows)
            self.show()

    def save(self, status):
        if not self.rows:
            return
        try:
            save_review(self.dataset, self.rows[self.position]['image_id'], self.food.get(), self.condition.get(), status,
                        self.reviewer.get(), self.evidence.get(), self.group.get())
            self.reload()
            if status != 'Pending':
                self.move(1)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Chua luu duoc', str(exc), parent=self)

    def sync(self):
        if not self.require_saved():
            return
        try:
            result = import_google(self.dataset, self.project / 'food15_workspace/image_downloads')
            self.reload()
            messagebox.showinfo('Nhap anh', f"Them {result['added']} anh. Loi: {len(result['errors'])}\n" + '\n'.join(result['errors'][:5]), parent=self)
        except Exception as exc:
            messagebox.showerror('Loi nhap anh', str(exc), parent=self)

    def open_source(self):
        if self.rows:
            source = self.dataset.sources()[self.rows[self.position]['source_dataset']]
            url = source.get('source_url', '')
            if url.startswith(('http://', 'https://')):
                webbrowser.open(url)

    def report(self):
        report = self.dataset.coverage()
        messagebox.showinfo('Do phu dataset',
            f"Muc tieu: {report['target_foods']} thuc pham / {report['target_classes']} lop\n"
            f"So lop co anh du dieu kien: {report['approved_classes']}\n"
            f"San sang train: {'Co' if report['ready_for_training'] else 'Chua'}\n"
            'Duyet nhan khong tu dong duyet giay phep. Xem reports/coverage.csv va readiness.json.', parent=self)

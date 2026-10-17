"""
FoodQuality-8 - Giao dien do hoa
==================================
Double-click Mo-GUI-FoodQuality.bat de mo.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, scrolledtext, ttk

PROJECT_DIR = Path(__file__).resolve().parent

def _find_python() -> str:
    candidates = [
        PROJECT_DIR / ".venv" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable

PYTHON_EXE = _find_python()

CLR_BG      = "#1E2128"
CLR_PANEL   = "#252931"
CLR_ACCENT  = "#4CAF82"
CLR_DANGER  = "#E05C5C"
CLR_WARN    = "#E8B84B"
CLR_BLUE    = "#5B9BD5"
CLR_TEXT    = "#E0E4EC"
CLR_SUBTEXT = "#8A8FA8"
CLR_LOG_BG  = "#161820"


# ─────────────────────────────────────────────────────────────────────────────
# Widget Log dung chung
# ─────────────────────────────────────────────────────────────────────────────
def make_log(parent, mono9) -> scrolledtext.ScrolledText:
    log = scrolledtext.ScrolledText(
        parent, state="disabled", font=mono9,
        bg=CLR_LOG_BG, fg="#A8D8A8", insertbackground=CLR_TEXT,
        relief="flat", bd=0, wrap="word")
    log.tag_config("info",    foreground="#A8D8A8")
    log.tag_config("warn",    foreground=CLR_WARN)
    log.tag_config("error",   foreground=CLR_DANGER)
    log.tag_config("success", foreground=CLR_ACCENT)
    log.tag_config("dim",     foreground=CLR_SUBTEXT)
    return log

def log_line(log: scrolledtext.ScrolledText, text: str, tag: str = "info"):
    log.configure(state="normal")
    log.insert("end", text, tag)
    log.see("end")
    log.configure(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class PipelineTab(tk.Frame):
    def __init__(self, parent, bold11, reg10, mono9):
        super().__init__(parent, bg=CLR_BG)
        self._process = None
        self._running = threading.Event()
        self._bold11 = bold11
        self._reg10  = reg10
        self._mono9  = mono9
        self._build()

    def _build(self):
        bold11, reg10, mono9 = self._bold11, self._reg10, self._mono9

        # Cau hinh
        cfg = tk.LabelFrame(self, text="  Cau hinh  ", font=bold11,
                             bg=CLR_BG, fg=CLR_TEXT, bd=1, relief="groove",
                             padx=12, pady=8)
        cfg.pack(fill="x", padx=12, pady=(10, 0))

        tk.Label(cfg, text="Script chay:", font=reg10,
                 bg=CLR_BG, fg=CLR_TEXT, width=14, anchor="w").grid(
                     row=0, column=0, sticky="w", pady=3)
        self._script_var = tk.StringVar(
            value=str(PROJECT_DIR / "scripts" / "food15_pipeline.py"))
        tk.Entry(cfg, textvariable=self._script_var, font=reg10,
                 bg=CLR_PANEL, fg=CLR_TEXT, insertbackground=CLR_TEXT,
                 relief="flat", bd=4).grid(row=0, column=1, sticky="ew", padx=(0,6))
        tk.Button(cfg, text="...", font=reg10, bg=CLR_PANEL, fg=CLR_TEXT,
                  relief="flat", cursor="hand2",
                  command=self._browse_script).grid(row=0, column=2)

        tk.Label(cfg, text="File Excel:", font=reg10,
                 bg=CLR_BG, fg=CLR_TEXT, width=14, anchor="w").grid(
                     row=1, column=0, sticky="w", pady=3)
        self._xlsx_var = tk.StringVar(value=str(
            PROJECT_DIR / "food15_workspace" / "reports" / "FoodQuality-15-5-review.xlsx"))
        tk.Entry(cfg, textvariable=self._xlsx_var, font=reg10,
                 bg=CLR_PANEL, fg=CLR_TEXT, insertbackground=CLR_TEXT,
                 relief="flat", bd=4).grid(row=1, column=1, sticky="ew", padx=(0,6))
        tk.Button(cfg, text="...", font=reg10, bg=CLR_PANEL, fg=CLR_TEXT,
                  relief="flat", cursor="hand2",
                  command=self._browse_xlsx).grid(row=1, column=2)
        cfg.columnconfigure(1, weight=1)

        # Nut
        btn_f = tk.Frame(self, bg=CLR_BG)
        btn_f.pack(fill="x", padx=12, pady=10)
        bc = dict(font=bold11, relief="flat", cursor="hand2", padx=18, pady=8, bd=0)

        self._btn_run = tk.Button(btn_f, text="  CHAY", bg=CLR_ACCENT,
                                   fg="#0D1117", activebackground="#3B9966",
                                   command=self._run, **bc)
        self._btn_run.pack(side="left", padx=(0,8))

        self._btn_stop = tk.Button(btn_f, text="  NGUNG", bg=CLR_DANGER,
                                    fg="white", activebackground="#B04040",
                                    state="disabled", command=self._stop, **bc)
        self._btn_stop.pack(side="left", padx=(0,8))

        self._btn_view = tk.Button(btn_f, text="  XEM KET QUA (Excel)",
                                    bg=CLR_WARN, fg="#0D1117",
                                    activebackground="#C09030",
                                    command=self._view_results, **bc)
        self._btn_view.pack(side="left")
        tk.Button(btn_f, text='XUAT DATASET TRAIN', bg=CLR_BLUE, fg='white',
                  command=self._export_training, font=reg10, relief='flat', padx=8, pady=8).pack(side='left', padx=8)

        self._status_var = tk.StringVar(value="San sang")
        self._status_lbl = tk.Label(btn_f, textvariable=self._status_var,
                                     font=reg10, bg=CLR_BG, fg=CLR_SUBTEXT)
        self._status_lbl.pack(side="right", padx=6)

        # Progress
        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill="x", padx=12)

        # Log
        tk.Label(self, text="  Nhat ky", font=bold11,
                 bg=CLR_BG, fg=CLR_TEXT, anchor="w").pack(
                     fill="x", padx=12, pady=(8,0))
        self._log = make_log(self, mono9)
        self._log.pack(fill="both", expand=True, padx=12, pady=4)
        log_line(self._log, "Pipeline san sang. Nhan CHAY de bat dau.\n", "dim")

    def _browse_script(self):
        p = filedialog.askopenfilename(
            title="Chon script Python",
            initialdir=str(PROJECT_DIR / "scripts"),
            filetypes=[("Python", "*.py"), ("Tat ca", "*.*")])
        if p:
            self._script_var.set(p)

    def _export_training(self):
        if self._process or self._running.is_set():
            return
        from dataset_export_gui import ExportWindow
        window=ExportWindow(self,PROJECT_DIR)
        window.grab_set()

    def _browse_xlsx(self):
        p = filedialog.askopenfilename(
            title="Chon file Excel ket qua",
            initialdir=str(PROJECT_DIR),
            filetypes=[("Excel", "*.xlsx *.xls"), ("Tat ca", "*.*")])
        if p:
            self._xlsx_var.set(p)

    def _log_safe(self, text, tag="info"):
        self.after(0, log_line, self._log, text, tag)

    def _run(self):
        script = self._script_var.get().strip()
        if not script or not Path(script).is_file():
            messagebox.showerror("Loi", f"Khong tim thay script:\n{script}")
            return
        self._btn_run.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._progress.start(12)
        self._status_var.set("Dang chay...")
        self._status_lbl.config(fg=CLR_ACCENT)
        self._running.set()
        self._log_safe("-" * 60 + "\n", "dim")
        self._log_safe(f"Bat dau: {Path(script).name}\n", "success")
        threading.Thread(target=self._worker, args=(script,), daemon=True).start()

    def _worker(self, script):
        rc = -1
        try:
            self._process = subprocess.Popen(
                [PYTHON_EXE, script], cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in self._process.stdout:
                if not self._running.is_set():
                    break
                tag = ("error" if any(w in line.lower() for w in
                       ("error","exception","traceback")) else
                       "warn" if any(w in line.lower() for w in
                       ("warning","warn","skip")) else "info")
                self._log_safe(line, tag)
            self._process.wait()
            rc = self._process.returncode
        except Exception as exc:
            self._log_safe(f"[Loi he thong] {exc}\n", "error")
        finally:
            self._process = None
        if rc == 0:
            self._log_safe("Hoan thanh!\n", "success")
            self.after(0, self._on_done, True)
        else:
            msg = "Da ngung.\n" if not self._running.is_set() else f"That bai (exit {rc})\n"
            self._log_safe(msg, "warn")
            self.after(0, self._on_done, False)

    def _on_done(self, success):
        self._running.clear()
        self._progress.stop()
        self._btn_run.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_var.set("Hoan thanh" if success else "Da dung / loi")
        self._status_lbl.config(fg=CLR_ACCENT if success else CLR_SUBTEXT)

    def _stop(self):
        self._running.clear()
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass
        self._log_safe("Yeu cau ngung da gui...\n", "warn")
        self._btn_stop.config(state="disabled")

    def _view_results(self):
        xlsx = Path(self._xlsx_var.get().strip())
        if not xlsx.is_file():
            messagebox.showwarning(
                "Chua co ket qua",
                f"Khong tim thay file:\n{xlsx}\n\nHay chay pipeline truoc.")
            return
        os.startfile(str(xlsx))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Tim kiem anh thuc pham
# ─────────────────────────────────────────────────────────────────────────────
class SearchTab(tk.Frame):
    def __init__(self, parent, bold11, reg10, mono9):
        super().__init__(parent, bg=CLR_BG)
        self._bold11 = bold11
        self._reg10  = reg10
        self._mono9  = mono9
        self._stop_event = threading.Event()
        self._running = False

        # Mac dinh duong dan Excel ket qua tim kiem
        self._out_xlsx = PROJECT_DIR / "food15_workspace" / "reports" / "ImageSearch.xlsx"
        self._downloads_root = PROJECT_DIR / "food15_workspace" / "image_downloads"

        self._build()

    def _build(self):
        bold11, reg10, mono9 = self._bold11, self._reg10, self._mono9

        # ── Thanh tim kiem ───────────────────────────────────────────────────
        search_f = tk.LabelFrame(self, text="  Tìm ảnh minh họa · Tươi / Hỏng  ",
                                  font=bold11, bg=CLR_BG, fg=CLR_TEXT,
                                  bd=1, relief="groove", padx=12, pady=10)
        search_f.pack(fill="x", padx=12, pady=(10, 0))

        # Dong 1: Nhap ten thuc pham
        tk.Label(search_f, text="Tên thực phẩm:", font=bold11,
                 bg=CLR_BG, fg=CLR_TEXT, anchor="w").grid(
                     row=0, column=0, sticky="w", pady=4)

        self._food_var = tk.StringVar()
        food_entry = tk.Entry(search_f, textvariable=self._food_var,
                               font=tkfont.Font(family="Segoe UI", size=13),
                               bg=CLR_PANEL, fg=CLR_TEXT,
                               insertbackground=CLR_TEXT,
                               relief="flat", bd=6)
        food_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        food_entry.bind("<Return>", lambda e: self._start_search())

        tk.Label(search_f, text="Số ảnh (0 = liên tục):", font=reg10,
                 bg=CLR_BG, fg=CLR_TEXT).grid(row=0, column=2, sticky="e")
        self._num_var = tk.IntVar(value=90)
        tk.Spinbox(search_f, from_=0, to=100000, increment=50,
                   textvariable=self._num_var, width=5,
                   font=reg10, bg=CLR_PANEL, fg=CLR_TEXT,
                   buttonbackground=CLR_PANEL, relief="flat",
                   ).grid(row=0, column=3, padx=(4,0))

        search_f.columnconfigure(1, weight=1)

        # Dong 2: Duong dan Excel ket qua
        tk.Label(search_f, text="File Excel xuat:", font=reg10,
                 bg=CLR_BG, fg=CLR_TEXT, anchor="w").grid(
                     row=1, column=0, sticky="w", pady=(4,0))
        self._xlsx_var = tk.StringVar(value=str(self._out_xlsx))
        tk.Entry(search_f, textvariable=self._xlsx_var, font=reg10,
                 bg=CLR_PANEL, fg=CLR_TEXT, insertbackground=CLR_TEXT,
                 relief="flat", bd=4).grid(
                     row=1, column=1, columnspan=2, sticky="ew", padx=(8,8))
        tk.Button(search_f, text="...", font=reg10, bg=CLR_PANEL,
                  fg=CLR_TEXT, relief="flat", cursor="hand2",
                  command=self._browse_xlsx).grid(row=1, column=3)

        # ── Nut dieu khien ───────────────────────────────────────────────────
        btn_f = tk.Frame(self, bg=CLR_BG)
        btn_f.pack(fill="x", padx=12, pady=10)
        bc = dict(font=bold11, relief="flat", cursor="hand2", padx=18, pady=8, bd=0)

        self._btn_search = tk.Button(btn_f, text="TÌM VÀ LỌC ẢNH",
                                      bg=CLR_BLUE, fg="white",
                                      activebackground="#3A7ABF",
                                      command=self._start_search, **bc)
        self._btn_search.pack(side="left", padx=(0,8))

        self._btn_stop = tk.Button(btn_f, text="NGỪNG", bg=CLR_DANGER,
                                    fg="white", activebackground="#B04040",
                                    state="disabled",
                                    command=self._stop_search, **bc)
        self._btn_stop.pack(side="left", padx=(0,8))

        self._btn_open = tk.Button(btn_f, text="XEM EXCEL",
                                    bg=CLR_WARN, fg="#0D1117",
                                    activebackground="#C09030",
                                    command=self._open_excel, **bc)
        self._btn_open.pack(side="left")
        tk.Button(btn_f,text='HUẤN LUYỆN',bg=CLR_ACCENT,fg='#0D1117',command=self._train_images,**bc).pack(side='left',padx=8)

        self._status_var = tk.StringVar(value="San sang tim kiem")
        self._status_lbl = tk.Label(btn_f, textvariable=self._status_var,
                                     font=reg10, bg=CLR_BG, fg=CLR_SUBTEXT)
        self._status_lbl.pack(side="right", padx=6)

        # Progress
        style = ttk.Style()
        style.configure("P2.Horizontal.TProgressbar", troughcolor=CLR_PANEL,
                        background=CLR_BLUE, thickness=6)
        self._progress = ttk.Progressbar(self, mode="indeterminate",
                                          style="P2.Horizontal.TProgressbar")
        self._progress.pack(fill="x", padx=12)

        # Goi y nhanh
        self._auto_train=tk.BooleanVar(value=True)
        tk.Checkbutton(self,text='Tự huấn luyện cả hai mô hình khi tìm đủ ảnh',variable=self._auto_train,
            bg=CLR_BG,fg=CLR_TEXT,selectcolor=CLR_PANEL,activebackground=CLR_BG).pack(anchor='w',padx=12,pady=4)
        hint_f = tk.Frame(self, bg=CLR_BG)
        hint_f.pack(fill="x", padx=12, pady=(6,0))
        tk.Label(hint_f, text="Goi y nhanh:", font=reg10,
                 bg=CLR_BG, fg=CLR_SUBTEXT).pack(side="left")
        for food in ["táo", "chuối", "thịt bò", "thịt gà", "cà chua", "thịt lợn"]:
            tk.Button(hint_f, text=food, font=reg10,
                      bg=CLR_PANEL, fg=CLR_TEXT, relief="flat",
                      cursor="hand2", padx=6, pady=2,
                      command=lambda f=food: self._quick_fill(f)
                      ).pack(side="left", padx=3)

        # Log
        tk.Label(self, text="  Nhat ky tim kiem", font=bold11,
                 bg=CLR_BG, fg=CLR_TEXT, anchor="w").pack(
                     fill="x", padx=12, pady=(8,0))
        self._log = make_log(self, mono9)
        self._log.pack(fill="both", expand=True, padx=12, pady=4)
        log_line(self._log,
                 "Nhap ten thuc pham va nhan TIM KIEM ANH.\n"
                 "So anh = 0: chay den khi NGUNG. So > 0: tong so anh can co cho loai nay.\n"
                 "Edge se mo Google; neu can xac minh, hay tu thao tac trong Edge.\n"
                 "Tự tạo lệnh: Tươi -> Hỏng (Việt/Anh).\n"
                 "NGUNG van luu anh da tai vao Excel, moi loai 1 sheet.\n"
                 "Nhãn/vùng được đề xuất tự động; sửa nhãn là tùy chọn.\n",
                 "dim")

    def _quick_fill(self, food: str):
        self._food_var.set(food)

    def _train_images(self):
        if self._running:
            messagebox.showinfo('Đang tìm ảnh','Bấm Ngừng hoặc đợi tìm xong trước khi huấn luyện.')
            return
        app=self.winfo_toplevel()
        if getattr(app,'_closing',False):
            return
        tab=app._training_tab
        tab.food_name.set(self._food_var.get().strip())
        tab.source.set('google')
        tab.excel=self._xlsx_var.get()
        app._notebook.select(tab)
        tab.start()

    def _browse_xlsx(self):
        p = filedialog.asksaveasfilename(
            title="Chon file Excel xuat ket qua",
            initialdir=str(PROJECT_DIR / "food15_workspace" / "reports"),
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Tat ca", "*.*")])
        if p:
            self._xlsx_var.set(p)

    def _log_safe(self, text, tag="info"):
        self.after(0, log_line, self._log, text, tag)

    def _start_search(self):
        food_name = self._food_var.get().strip()
        if not food_name:
            messagebox.showwarning("Thieu thong tin",
                                   "Vui long nhap ten thuc pham truoc khi tim kiem.")
            return
        if self._running:
            return

        try:
            max_num = self._num_var.get()
            if max_num < 0:
                raise ValueError()
        except (ValueError, tk.TclError):
            messagebox.showwarning("So anh", "Nhap so nguyen >= 0; 0 = chay den khi bam NGUNG.")
            return
        if not self._xlsx_var.get().strip().lower().endswith('.xlsx'):
            messagebox.showwarning("File Excel", "Chon duong dan ket qua co duoi .xlsx.")
            return

        self._stop_event.clear()
        self._running = True
        self._btn_search.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._progress.start(10)
        self._status_var.set(f"Dang tim: {food_name}...")
        self._status_lbl.config(fg=CLR_BLUE)

        xlsx_path = Path(self._xlsx_var.get().strip())
        downloads_root = self._downloads_root

        threading.Thread(
            target=self._search_worker,
            args=(food_name, max_num, downloads_root, xlsx_path),
            daemon=True,
        ).start()

    def _search_worker(self, food_name, max_num, downloads_root, xlsx_path):
        try:
            # Import o day de tranh loi neu chua cai
            from image_search import search_and_export
            result_path = search_and_export(
                food_name=food_name,
                max_num=max_num,
                downloads_root=downloads_root,
                excel_path=xlsx_path,
                stop_event=self._stop_event,
                log_cb=lambda msg: self._log_safe(msg + "\n"),
            )
            from foodquality8.quick_dataset import prepare_google
            from image_search import food_key
            self._log_safe('Đang lọc ảnh gần trùng và chuẩn bị dữ liệu huấn luyện...\n')
            try:
                release=prepare_google(PROJECT_DIR,food_name,result_path)
                self._log_safe(f'Bộ dữ liệu thử nghiệm đã sẵn sàng: {release}\n','success')
            except ValueError as exc:
                self._log_safe(str(exc)+'\n','warn')
            report=PROJECT_DIR/'food15_workspace/image_downloads'/food_key(food_name)/'filter-report.json'
            if report.exists():
                result_path=json.loads(report.read_text(encoding='utf8')).get('excel_path',result_path)
            (PROJECT_DIR/'food15_workspace/last_search.json').write_text(json.dumps(dict(food_name=food_name,excel_path=result_path),ensure_ascii=False),encoding='utf8')
            if not self._stop_event.is_set():
                self._log_safe(f"\nHoan thanh! File: {result_path}\n", "success")
                self.after(0, self._on_done, True, result_path)
            else:
                self._log_safe(f"Da ngung va luu anh da tai: {result_path}\n", "success")
                self.after(0, self._on_done, False, result_path)
        except Exception as exc:
            self._log_safe(f"\n[Loi] {exc}\n", "error")
            self.after(0, self._on_done, False, "")

    def _on_done(self, success, result_path=""):
        self._running = False
        if result_path:
            self._xlsx_var.set(result_path)
            tab=self.winfo_toplevel()._training_tab
            tab.food_name.set(self._food_var.get())
            tab.source.set('google')
            tab.excel=result_path
        self._progress.stop()
        self._btn_search.config(state="normal")
        self._btn_stop.config(state="disabled")
        if success:
            self._status_var.set("Hoan thanh!")
            self._status_lbl.config(fg=CLR_ACCENT)
            self._log_safe('Đã xuất Excel. Bấm HUẤN LUYỆN để tự lọc, đề xuất vùng và chạy lần lượt cả hai mô hình.\n','success')
            if self._auto_train.get():
                self.after(100,self._train_images)
        else:
            self._status_var.set("Da dung - da luu Excel" if result_path else "Da dung / loi")
            self._status_lbl.config(fg=CLR_SUBTEXT)

    def _stop_search(self):
        self._stop_event.set()
        self._log_safe("Dang dung tim kiem...\n", "warn")
        self._btn_stop.config(state="disabled")

    def _open_excel(self):
        xlsx = Path(self._xlsx_var.get().strip())
        if not xlsx.is_file():
            messagebox.showwarning(
                "Chua co ket qua",
                f"Khong tim thay file:\n{xlsx}\n\nHay tim kiem truoc.")
            return
        os.startfile(str(xlsx))


# ─────────────────────────────────────────────────────────────────────────────
# Cua so chinh
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FoodQuality - Dataset va hai model AI")
        self.configure(bg=CLR_BG)
        self.geometry("900x660")
        self.minsize(760, 520)
        self.resizable(True, True)
        try:
            self.iconbitmap(str(PROJECT_DIR / "icon.ico"))
        except Exception:
            pass
        self._build()
        self.protocol('WM_DELETE_WINDOW', self._close)

    def _close(self):
        self._closing=True
        if self._search_tab._running:
            if not self._search_tab._stop_event.is_set():
                self._search_tab._stop_search()
            self.after(300, self._close)
            return
        waiting = False
        for tab in self._model_tabs:
            if tab.preparing or (tab.process and tab.process.poll() is None):
                tab.stop()
                waiting = True
        if waiting:
            self.after(500, self._close)
            return
        self.destroy()

    def _build(self):
        bold14 = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        bold11 = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        reg10  = tkfont.Font(family="Segoe UI", size=10)
        mono9  = tkfont.Font(family="Consolas", size=9)

        # Header
        header = tk.Frame(self, bg=CLR_PANEL, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="FoodQuality", font=bold14,
                 bg=CLR_PANEL, fg=CLR_ACCENT).pack(side="left", padx=18)
        tk.Label(header, text="Thu thập dữ liệu và huấn luyện AI",
                 font=reg10, bg=CLR_PANEL, fg=CLR_SUBTEXT).pack(side="left")
        tk.Button(header, text='SỬA NHÃN (TÙY CHỌN)', command=self._open_review,
                  bg=CLR_WARN, fg='#0D1117', relief='flat', padx=10, pady=6).pack(side='right', padx=12)

        # Notebook (tabs)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=CLR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=CLR_PANEL,
                        foreground=CLR_SUBTEXT, padding=[14, 6],
                        font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", CLR_BG)],
                  foreground=[("selected", CLR_TEXT)])

        nb = ttk.Notebook(self)
        self._notebook=nb
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        tab2 = SearchTab(nb, bold11, reg10, mono9)
        self._search_tab = tab2
        nb.add(tab2, text="  1. Tìm ảnh  ")
        from simple_training_gui import TrainingTab
        self._training_tab=TrainingTab(nb,PROJECT_DIR,PYTHON_EXE)
        self._model_tabs=[self._training_tab]
        nb.add(self._training_tab,text='  2. Huấn luyện AI  ')
        nb.select(tab2)

        # Footer
        footer = tk.Frame(self, bg=CLR_PANEL, pady=3)
        footer.pack(fill="x", side="bottom")
        tk.Label(footer, text=f"Python: {PYTHON_EXE}",
                 font=("Consolas", 8), bg=CLR_PANEL, fg=CLR_SUBTEXT
                 ).pack(side="left", padx=10)
        tk.Button(footer, text="Mở giao diện người dùng", command=self._open_user_app,
                 font=("Consolas", 8), bg=CLR_PANEL, fg=CLR_SUBTEXT
                 ).pack(side="right", padx=10)

    def _open_user_app(self):
        subprocess.Popen([PYTHON_EXE, str(PROJECT_DIR/'launch_user_app.py')], cwd=PROJECT_DIR,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def _open_review(self):
        if self._notebook.select()==str(self._search_tab):
            self._training_tab.food_name.set(self._search_tab._food_var.get())
            self._training_tab.source.set('google')
        self._training_tab.review()


if __name__ == "__main__":
    app = App()
    app.mainloop()

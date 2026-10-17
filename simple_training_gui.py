"""Minimal controls for the complete two-model workflow."""
import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext


class TrainingTab(ttk.Frame):
    def __init__(self,parent,project,python):
        super().__init__(parent,padding=16)
        self.project,self.python=project,str(python)
        self.process=None
        self.preparing=False
        self.run_dir=None
        self.events=queue.Queue()
        self.food_name=tk.StringVar()
        self.source=tk.StringVar(value='Dataset kết hợp SIC' if (project/'models/fusion.json').exists() else 'google')
        self.excel=None
        self.epochs=tk.IntVar(value=20)
        self.status=tk.StringVar(value='Chọn ảnh có sẵn hoặc dùng kết quả Google. Một lần bấm chạy cả hai mô hình.')
        ttk.Label(self,text='Huấn luyện hai mô hình tự động',font=('Segoe UI',16,'bold')).pack(anchor='w',pady=8)
        form=ttk.Frame(self)
        form.pack(fill='x',pady=8)
        ttk.Label(form,text='Thực phẩm (Google/ảnh riêng):').grid(row=0,column=0,sticky='w')
        ttk.Entry(form,textvariable=self.food_name,width=30).grid(row=0,column=1,sticky='ew',padx=8)
        ttk.Label(form,text='Nguồn dữ liệu:').grid(row=1,column=0,sticky='w',pady=10)
        ttk.Combobox(form,textvariable=self.source,values=['Dataset kết hợp SIC','google']).grid(row=1,column=1,sticky='ew',padx=8)
        form.columnconfigure(1,weight=1)
        picks=ttk.Frame(self)
        picks.pack(fill='x')
        ttk.Button(picks,text='Ảnh Google',command=lambda:self.source.set('google')).pack(side='left')
        ttk.Button(picks,text='Chọn thư mục ảnh',command=self.folder).pack(side='left',padx=5)
        ttk.Button(picks,text='Chọn Excel',command=self.workbook).pack(side='left')
        ttk.Label(self,text='Ảnh có sẵn: chọn thư mục có các thư mục con Tuoi / Hong hoặc Excel có ảnh và chất lượng.\n'
            'Dataset kết hợp SIC: tự tạo biến thể, học 9 loại với hai vùng 320×320; giữ YOLO SIC, kiểm tra trước khi cập nhật website.\n'
            'Nhãn/vùng tự động chưa được kiểm chứng; kết quả website sẽ ghi rõ chế độ thử nghiệm.',wraplength=840).pack(anchor='w',pady=12)
        actions=ttk.Frame(self)
        actions.pack(fill='x')
        self.train=ttk.Button(actions,text='HUẤN LUYỆN AI',command=self.start)
        self.train.pack(side='left')
        ttk.Button(actions,text='NGỪNG',command=self.stop).pack(side='left',padx=6)
        ttk.Button(actions,text='XEM KẾT QUẢ',command=self.results).pack(side='left')
        ttk.Label(actions,text='Tự chọn bản tốt trên tập validation; dừng sớm khi không cải thiện.').pack(side='left',padx=(15,4))
        ttk.Label(self,textvariable=self.status,wraplength=840).pack(fill='x',pady=12)
        self.log=scrolledtext.ScrolledText(self,state='disabled',height=15,font=('Consolas',9))
        self.log.pack(fill='both',expand=True)
        last=project/'food15_workspace/last_search.json'
        if last.exists():
            try:
                data=json.loads(last.read_text(encoding='utf8'))
                self.food_name.set(data.get('food_name',''))
                self.excel=data.get('excel_path')
            except (OSError,ValueError):
                pass
        self.after(150,self.poll)

    def folder(self):
        path=filedialog.askdirectory(title='Chọn thư mục chứa ảnh theo mức chất lượng')
        if path:
            self.source.set(path)

    def workbook(self):
        path=filedialog.askopenfilename(filetypes=[('Excel có ảnh','*.xlsx')])
        if path:
            self.source.set(path)

    def review(self):
        from simple_review_gui import SimpleReviewWindow
        if not self.food_name.get().strip():
            messagebox.showinfo('Tên thực phẩm','Nhập tên thực phẩm trước.',parent=self)
            return
        try:
            SimpleReviewWindow(self,self.project,self.food_name.get().strip(),self.source.get().strip())
        except Exception as exc:
            messagebox.showerror('Chưa mở được ảnh',str(exc),parent=self)

    def start(self):
        if self.process and self.process.poll() is None:
            return
        food=self.food_name.get().strip()
        source=self.source.get().strip()
        sic_source=source=='Dataset kết hợp SIC'
        if (not food and not sic_source) or not source:
            messagebox.showinfo('Thiếu dữ liệu','Nhập tên thực phẩm và chọn nguồn ảnh.',parent=self)
            return
        try:
            epochs=self.epochs.get()
            if not 1<=epochs<=500:
                raise ValueError()
        except (ValueError,tk.TclError):
            messagebox.showerror('Số vòng','Nhập số vòng từ 1 đến 500.',parent=self)
            return
        self.run_dir=self.project/'model_runs/pairs'/str(time.time_ns())
        self.run_dir.mkdir(parents=True)
        args=[self.python,'-u','-m','foodquality8.train_pair','--project',str(self.project),
            '--food',food,'--source',source,'--run',str(self.run_dir),'--epochs',str(epochs)]
        if source=='google' and self.excel:
            args+=['--excel',str(self.excel)]
        if sic_source:
            args=[self.python,'-X','utf8','-u',str(self.project/'scripts/retrain_robust.py'),'--run',str(self.run_dir)]
        try:
            self.process=subprocess.Popen(args,cwd=self.project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                env={**os.environ,'PYTHONUTF8':'1'})
        except OSError as exc:
            messagebox.showerror('Không chạy được',str(exc),parent=self)
            return
        self.train.configure(state='disabled')
        self.status.set('Đang huấn luyện và kiểm tra tự động. Dataset SIC dùng YOLO có sẵn; Google/ảnh riêng học cả phân loại và khoanh vùng.')
        process=self.process
        def read():
            for line in process.stdout:
                self.events.put(('log',line))
            self.events.put(('done',process.wait()))
        threading.Thread(target=read,daemon=True).start()

    def stop(self):
        if self.process and self.process.poll() is None and self.run_dir:
            (self.run_dir/'STOP').touch()
            self.status.set('Đang ngừng sau bước/lượt học hiện tại; không chạy mô hình tiếp theo.')

    def results(self):
        folder=self.run_dir or self.project/'model_runs/pairs'
        folder.mkdir(parents=True,exist_ok=True)
        report=folder/'report.html'
        os.startfile(report if report.exists() else folder)

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='log':
                    self.log.configure(state='normal')
                    self.log.insert('end',value)
                    self.log.see('end')
                    self.log.configure(state='disabled')
                else:
                    self.train.configure(state='normal')
                    state={}
                    path=self.run_dir/'workflow.json'
                    if path.exists():
                        state=json.loads(path.read_text(encoding='utf8'))
                    self.status.set(state.get('message') or (('Đã học xong. '+('Đã cập nhật mô hình minh họa; chọn chế độ Minh họa huấn luyện Google trên website để xem.' if state.get('activated_new_models') else 'Giữ mô hình minh họa cũ vì chưa có bằng chứng bản mới tốt hơn.')) if state.get('status')=='complete' else 'Chưa hoàn tất; xem nhật ký.'))
        except queue.Empty:
            pass
        self.after(150,self.poll)

"""Admin-only training controls, subprocess logs and model activation."""
import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext


class ModelTab(ttk.Frame):
    def __init__(self, parent, project, python, kind):
        super().__init__(parent, padding=12)
        self.project, self.python, self.kind = project, python, kind
        self.process = None
        self.checking = False
        self.run_dir = None
        self.preparing = False
        self.cancel_prepare = False
        self.events = queue.Queue()
        self.release = tk.StringVar(value=str(project/'food15_workspace/releases'))
        self.epochs = tk.StringVar(value='10')
        self.batch = tk.StringVar(value='4')
        self.device = tk.StringVar(value='auto')
        self.quick=tk.BooleanVar(value=kind=='classifier')
        self.food_name=tk.StringVar()
        last=project/'food15_workspace/last_search.json'
        if last.exists():
            self.food_name.set(json.loads(last.read_text(encoding='utf8')).get('food_name',''))
        title = 'Nhận diện và khoanh vùng · YOLO' if kind == 'detector' else 'Phân loại chất lượng · EfficientNet'
        ttk.Label(self, text=title, font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0,10))
        if kind=='classifier':
            ttk.Checkbutton(self,text='Dùng ảnh Google đã lọc — huấn luyện thử nghiệm với nhãn gợi ý',variable=self.quick,command=self.update_mode).pack(anchor='w',pady=4)
            quick_form=ttk.Frame(self)
            quick_form.pack(fill='x',pady=4)
            ttk.Label(quick_form,text='Thực phẩm đã tìm:').pack(side='left')
            ttk.Entry(quick_form,textvariable=self.food_name,width=25).pack(side='left',padx=8)
            ttk.Label(self,text='Bấm Huấn luyện: tự lọc, đọc sửa đổi trong Excel và chia tập. Điểm thử nghiệm chưa phải độ chính xác đã kiểm chứng.',wraplength=820).pack(anchor='w',pady=4)
        else:
            ttk.Label(self,text='YOLO cần vùng khoanh đã duyệt. Dùng nút Kiểm tra ảnh để khoanh vùng và xuất bộ dữ liệu.',wraplength=820).pack(anchor='w',pady=4)
        form = ttk.Frame(self)
        self.release_form=form
        form.pack(fill='x')
        ttk.Label(form, text='Bộ dữ liệu đã xuất:').grid(row=0,column=0,sticky='w')
        ttk.Entry(form,textvariable=self.release).grid(row=0,column=1,sticky='ew',padx=8)
        ttk.Button(form,text='Chọn thư mục',command=self.browse).grid(row=0,column=2)
        form.columnconfigure(1,weight=1)
        parameters = ttk.Frame(self)
        parameters.pack(fill='x',pady=10)
        for label,var in [('Số vòng học',self.epochs),('Ảnh mỗi lượt',self.batch)]:
            ttk.Label(parameters,text=label).pack(side='left',padx=4)
            ttk.Entry(parameters,textvariable=var,width=7).pack(side='left')
        ttk.Label(parameters,text='Thiết bị: tự động').pack(side='left',padx=8)
        controls = ttk.Frame(self)
        controls.pack(fill='x',pady=4)
        self.buttons = []
        for text, command in [('HUẤN LUYỆN',lambda:self.start(False)),('NGỪNG',self.stop),('XEM KẾT QUẢ',self.results)]:
            button = ttk.Button(controls,text=text,command=command)
            button.pack(side='left',padx=3)
            self.buttons.append(button)
        self.activate_button=ttk.Button(self,text='Dùng mô hình đã kiểm chứng cho ứng dụng',command=self.activate)
        self.activate_button.pack(anchor='w',pady=4)
        self.status = tk.StringVar(value='Sẵn sàng. Hệ thống tự kiểm tra dữ liệu khi bấm Huấn luyện.')
        ttk.Label(self,textvariable=self.status,wraplength=850).pack(fill='x',pady=8)
        self.log = scrolledtext.ScrolledText(self,height=18,state='disabled',font=('Consolas',9))
        self.log.pack(fill='both',expand=True)
        self.after(150,self.poll)
        self.update_mode()

    def update_mode(self):
        if self.quick.get() and self.kind=='classifier':
            self.release_form.pack_forget()
            self.activate_button.pack_forget()
        elif not self.release_form.winfo_manager():
            self.release_form.pack(fill='x',before=self.log)
            self.activate_button.pack(anchor='w',pady=4,before=self.log)

    def browse(self):
        folder = filedialog.askdirectory(initialdir=self.release.get())
        if folder:
            self.release.set(folder)

    def start(self, check):
        if self.preparing or (self.process and self.process.poll() is None):
            return
        try:
            epochs,batch=int(self.epochs.get()),int(self.batch.get())
            if not 1<=epochs<=500 or not 1<=batch<=128:
                raise ValueError()
        except ValueError:
            messagebox.showerror('Thông số','Số vòng học: 1..500; ảnh mỗi lượt: 1..128.')
            return
        if self.kind=='classifier' and self.quick.get():
            food=self.food_name.get().strip()
            if not food:
                messagebox.showwarning('Chưa chọn thực phẩm','Nhập tên thực phẩm đã tìm hoặc tìm ảnh trước.')
                return
            self.preparing=True
            self.cancel_prepare=False
            self.buttons[0].configure(state='disabled')
            self.status.set('Đang lọc ảnh, đọc Excel và chia bộ dữ liệu thử nghiệm...')
            def prepare():
                try:
                    from foodquality8.quick_dataset import prepare_google
                    excel=self.project/'food15_workspace/reports/ImageSearch.xlsx'
                    last=self.project/'food15_workspace/last_search.json'
                    if last.exists():
                        saved=json.loads(last.read_text(encoding='utf8'))
                        if saved.get('food_name')==food:
                            excel=Path(saved.get('excel_path',str(excel)))
                    release=prepare_google(self.project,food,excel)
                    self.events.put(('prepared',(release,epochs,batch)))
                except Exception as exc:
                    self.events.put(('prepare_error',str(exc)))
            threading.Thread(target=prepare,daemon=True).start()
            return
        self.launch(check,epochs,batch,self.release.get())

    def launch(self,check,epochs,batch,release,experimental=False):
        run = self.project/'model_runs'/self.kind/(time.strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns())[-6:])
        self.checking = check
        args=[self.python,'-u','-m','foodquality8.training',self.kind,'--release',release,
              '--run',str(run),'--epochs',str(epochs),'--batch',str(batch),'--device',self.device.get()]
        if check:
            args.append('--check')
        else:
            self.run_dir=run
        if experimental:
            args.append('--experimental')
        try:
            self.process=subprocess.Popen(args,cwd=self.project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                env={**os.environ,'PYTHONUTF8':'1'})
        except OSError as exc:
            messagebox.showerror('Khong chay duoc',str(exc))
            return
        self.status.set('Đang kiểm tra...' if check else 'Đang huấn luyện...')
        self.buttons[0].configure(state='disabled')
        process=self.process
        def read():
            for line in process.stdout:
                self.events.put(('log',line))
            self.events.put(('done',process.wait()))
        threading.Thread(target=read,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='prepared':
                    self.preparing=False
                    release,epochs,batch=value
                    self.release.set(release)
                    if self.cancel_prepare:
                        self.buttons[0].configure(state='normal')
                        self.status.set('Đã ngừng. Bộ dữ liệu đã lọc được giữ lại; chưa chạy huấn luyện.')
                    else:
                        self.launch(False,epochs,batch,release,experimental=True)
                elif kind=='prepare_error':
                    self.preparing=False
                    self.buttons[0].configure(state='normal')
                    self.status.set(value)
                    if not self.cancel_prepare:
                        messagebox.showwarning('Chưa đủ dữ liệu',value,parent=self)
                elif kind=='log':
                    self.log.configure(state='normal')
                    self.log.insert('end',value)
                    self.log.see('end')
                    self.log.configure(state='disabled')
                else:
                    self.status.set('Đã kết thúc. Bấm Xem kết quả.' if value==0 else 'Chưa hoàn thành; xem lý do trong nhật ký.')
                    self.buttons[0].configure(state='normal')
        except queue.Empty:
            pass
        self.after(150,self.poll)

    def stop(self):
        if self.preparing:
            self.cancel_prepare=True
            self.status.set('Đang hoàn tất lưu dữ liệu; sẽ không bắt đầu huấn luyện.')
            return
        if self.process and self.process.poll() is None:
            if not self.checking and self.run_dir and (self.run_dir/'run.json').is_file():
                (self.run_dir/'STOP').touch()
                self.status.set('Đã yêu cầu ngừng. Hệ thống lưu kết quả sau lượt học hiện tại.')
            else:
                self.process.terminate()
                self.status.set('Da dung kiem tra / khoi tao.')

    def results(self):
        if self.run_dir and self.run_dir.exists():
            report=self.run_dir/'report.html'
            os.startfile(report if report.exists() else self.run_dir)
        else:
            folder=self.project/'model_runs'/self.kind
            folder.mkdir(parents=True,exist_ok=True)
            os.startfile(folder)

    def activate(self):
        from foodquality8.inference import activate_run
        folder=filedialog.askdirectory(title='Chon luot train hoan tat (co run.json)',
                                       initialdir=str(self.project/'model_runs'/self.kind))
        if not folder:
            return
        try:
            data=json.loads((Path(folder)/'run.json').read_text(encoding='utf8'))
            if data.get('kind')!=self.kind:
                raise ValueError('Chon dung loai model cua tab nay.')
            activate_run(folder,self.project/'models/active.json')
            messagebox.showinfo('Da kich hoat','Model da duoc chon cho giao dien nguoi dung. Can du ca 2 model de phan tich.')
        except (OSError,ValueError,KeyError) as exc:
            messagebox.showerror('Chua kich hoat',str(exc))

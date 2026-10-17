import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

from foodquality8 import Dataset


def readiness_details(dataset):
    rows=dataset.rows()
    sources=dataset.sources()
    return dict(total=len(rows), unknown=sum(not dataset.valid_label(r['food_name'],r['condition']) for r in rows),
        pending=sum(r['review_status']!='Approved' for r in rows),
        source_pending=sum(sources[r['source_dataset']].get('license_status')!='Approved' for r in rows),
        eligible=sum(dataset.eligible(r) for r in rows),
        groups={label:len({r['group_id'] for r in rows if dataset.eligible(r) and r['food_name']+'_'+r['condition']==label})
                for label in dataset.classes})


class ExportWindow(tk.Toplevel):
    def __init__(self,parent,project):
        super().__init__(parent)
        # The caller may be a tab or the separate review window.
        self.app=parent
        while self.app.master is not None:
            self.app=self.app.master
        self.project=project
        self.title('Chon pham vi dataset train')
        self.geometry('820x700')
        self.events=queue.Queue()
        self.running=False
        self.protocol('WM_DELETE_WINDOW',self.close)
        ttk.Label(self,text='Chon cac lop cho ban train nay. Muc tieu 15 loai cua project duoc giu nguyen.\n'
                  'Moi lop can co anh da duyet nhan va nguon, du nhom doc lap cho train/val/test.',wraplength=660).pack(padx=12,pady=12)
        self.summary=tk.StringVar()
        ttk.Label(self,textvariable=self.summary,wraplength=780,foreground='#9b3b12').pack(fill='x',padx=12,pady=(0,8))
        self.listbox=tk.Listbox(self,selectmode='extended',exportselection=False,font=('Consolas',10))
        self.listbox.pack(fill='both',expand=True,padx=12)
        controls=ttk.Frame(self,padding=12)
        controls.pack(fill='x')
        ttk.Button(controls,text='Chon tat ca',command=lambda:self.listbox.selection_set(0,'end')).pack(side='left')
        ttk.Button(controls,text='Mo duyet anh',command=self.open_review).pack(side='left',padx=4)
        ttk.Button(controls,text='Cap nhat',command=self.refresh).pack(side='left',padx=4)
        self.run_button=ttk.Button(controls,text='Chia tap va xuat cac lop da chon',command=self.start)
        self.run_button.pack(side='left',padx=8)
        self.status=tk.StringVar(value='Giu Ctrl de chon nhieu lop. Can it nhat 2 lop cho EfficientNet.')
        ttk.Label(self,textvariable=self.status,wraplength=660).pack(fill='x',padx=12,pady=12)
        self.refresh()
        self.after(150,self.poll)

    def refresh(self):
        if self.running:
            return
        selected={self.classes[i] for i in self.listbox.curselection()} if hasattr(self,'classes') else set()
        d=Dataset(self.project/'food15_workspace')
        try:
            self.classes=list(d.classes)
            stats={s['label']:s for s in d.statistics()}
            details=readiness_details(d)
        finally:
            d.close()
        self.listbox.delete(0,'end')
        for i,label in enumerate(self.classes):
            self.listbox.insert('end',f"{label} | anh du dieu kien: {stats[label]['total']} | nhom: {details['groups'][label]}")
            if label in selected:
                self.listbox.selection_set(i)
        self.summary.set(f"Da nhap: {details['total']} anh | Du dieu kien: {details['eligible']} anh.\n"
            f"Chua co nhan hop le: {details['unknown']} | Chua duyet nhan: {details['pending']} | Chua duyet nguon: {details['source_pending']}.\n"
            'Cac ly do co the cung xuat hien tren mot anh. Anh da tai khong dong nghia voi anh san sang train.')
        self.run_button.configure(state='normal' if details['eligible'] else 'disabled')
        if not details['eligible']:
            self.status.set('Chua the xuat. Mo duyet anh -> gan nhan va can cu -> kiem tra quyen nguon -> Cap nhat. Khong tu gan nhan tu khoa Google.')

    def open_review(self):
        self.grab_release()
        self.app._open_review()

    def close(self):
        if self.running:
            messagebox.showinfo('Dang xuat','Cho ghi xong dataset truoc khi dong.',parent=self)
        else:
            self.destroy()

    def start(self):
        classes=[self.classes[i] for i in self.listbox.curselection()]
        if len(classes)<2:
            messagebox.showwarning('Chon lop','Chon it nhat 2 lop.',parent=self)
            return
        d=Dataset(self.project/'food15_workspace')
        try:
            details=readiness_details(d)
            missing=[label for label in classes if details['groups'][label]<3]
        finally:
            d.close()
        if missing:
            self.status.set('Chua the chia tap: '+str(len(missing))+' lop chua co it nhat 3 nhom anh doc lap du dieu kien. Bo lop chua co du lieu hoac bo sung/duyet anh. Vi du: '+', '.join(missing[:5]))
            return
        self.running=True
        self.run_button.configure(state='disabled')
        self.status.set('Dang chia tap theo nhom va xuat dataset...')
        def worker():
            d=Dataset(self.project/'food15_workspace')
            try:
                d.split(seed=42)
                d.coverage()
                stats={s['label']:s for s in d.statistics()}
                missing=[label for label in classes if not all(stats[label][s] for s in ('train','val','test'))]
                if missing:
                    raise ValueError('Sau khi chia theo nhom, cac lop sau van thieu train/val/test: '+', '.join(missing)+'. Can bo sung nhom anh doc lap; khong tach anh cung mau de bu so luong.')
                output=d.export('v'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'),selected_classes=classes)
                self.events.put(('ok',output))
            except Exception as exc:
                self.events.put(('error',str(exc)))
            finally:
                d.close()
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            state,value=self.events.get_nowait()
            self.running=False
            self.refresh()
            self.status.set(('Da xuat: ' if state=='ok' else 'Chua xuat: ')+value)
            if state=='ok':
                for tab in getattr(self.app,'_model_tabs',[]):
                    tab.release.set(value)
        except queue.Empty:
            pass
        self.after(150,self.poll)

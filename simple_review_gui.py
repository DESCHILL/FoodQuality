import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageOps, ImageTk

from foodquality8.model_data import write_json
from foodquality8.quick_dataset import STAGES


class SimpleReviewWindow(tk.Toplevel):
    def __init__(self,parent,project,food,source):
        from image_search import food_key
        from foodquality8.auto_workflow import import_local
        folder=project/'food15_workspace/image_downloads'/food_key(food) if source=='google' else import_local(project,source,food)
        rows=[json.loads(line) for line in (folder/'manifest.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
        if not rows:
            raise ValueError('Chưa có ảnh để sửa nhãn.')
        super().__init__(parent)
        self.title('Sửa nhãn tùy chọn')
        self.geometry('760x650')
        self.rows,self.folder,self.position=rows,folder,0
        self.edits=folder/'corrections.json'
        self.corrections=json.loads(self.edits.read_text(encoding='utf8')) if self.edits.exists() else {}
        self.picture=ttk.Label(self,anchor='center')
        self.picture.pack(fill='both',expand=True,padx=12,pady=12)
        ttk.Label(self,text='Tên thực phẩm: '+food,font=('Segoe UI',12,'bold')).pack(pady=6)
        self.grade=tk.StringVar()
        ttk.Label(self,text='Chất lượng:').pack()
        ttk.Combobox(self,textvariable=self.grade,values=list(STAGES.values()),state='readonly').pack(pady=6)
        self.status=tk.StringVar()
        ttk.Label(self,textvariable=self.status).pack(pady=6)
        actions=ttk.Frame(self)
        actions.pack(pady=12)
        ttk.Button(actions,text='Ảnh trước',command=lambda:self.move(-1)).pack(side='left',padx=4)
        ttk.Button(actions,text='Lưu và tiếp',command=self.save).pack(side='left',padx=4)
        ttk.Button(actions,text='Bỏ ảnh này',command=self.exclude).pack(side='left',padx=4)
        self.show()

    def show(self):
        row=self.rows[self.position]
        edit=self.corrections.get(row['food_code'],{})
        self.grade.set(STAGES.get(edit.get('stage',row.get('search_stage')),''))
        try:
            with Image.open(row['img_path']) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((700,430))
                self.photo=ImageTk.PhotoImage(image)
            self.picture.configure(image=self.photo,text='')
        except OSError:
            self.picture.configure(image='',text='Ảnh không đọc được')
        self.status.set(f'Ảnh {self.position+1}/{len(self.rows)}'+(' · Đã bỏ' if edit.get('excluded') else ''))

    def move(self,step):
        self.position=(self.position+step)%len(self.rows)
        self.show()

    def save(self):
        stage=next((k for k,v in STAGES.items() if v==self.grade.get()),None)
        if stage:
            self.corrections[self.rows[self.position]['food_code']]=dict(stage=stage,excluded=False)
            write_json(self.edits,self.corrections)
            self.move(1)

    def exclude(self):
        self.corrections[self.rows[self.position]['food_code']]=dict(excluded=True)
        write_json(self.edits,self.corrections)
        self.move(1)

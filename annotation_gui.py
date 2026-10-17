import json
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image,ImageOps,ImageTk
from foodquality8.annotations import box_from_drag,save_boxes,save_source_review


class BoxWindow(tk.Toplevel):
    def __init__(self,parent,row,reviewer):
        super().__init__(parent)
        self.parent,self.row=parent,row
        self.title('Khoanh vung thuc pham - YOLO')
        self.reviewer=tk.StringVar(value=reviewer)
        self.boxes=json.loads(row['bbox_json'])
        ttk.Label(self,text='Keo chuot quanh MỘT thuc pham. Chon lai se thay vung cu. Ten: '+row['food_name']).pack(padx=12,pady=8)
        with Image.open(parent.dataset.root/row['raw_path']) as original:
            display=ImageOps.exif_transpose(original).convert('RGB')
            display.thumbnail((900,480))
            self.w,self.h=display.size
            self.photo=ImageTk.PhotoImage(display)
        self.canvas=tk.Canvas(self,width=self.w,height=self.h,highlightthickness=0,cursor='crosshair')
        self.canvas.pack(padx=12,pady=4)
        self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.start=None
        self.canvas.bind('<ButtonPress-1>',self.begin)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',self.end)
        footer=ttk.Frame(self,padding=10)
        footer.pack(fill='x')
        ttk.Label(footer,text='Nguoi duyet:').pack(side='left')
        ttk.Entry(footer,textvariable=self.reviewer,width=20).pack(side='left',padx=6)
        ttk.Button(footer,text='Luu va duyet box',command=self.save).pack(side='left',padx=6)
        ttk.Button(footer,text='Xoa box da luu',command=self.clear).pack(side='left')
        self.draw_saved()

    def draw_saved(self):
        self.canvas.delete('box')
        for b in self.boxes:
            self.canvas.create_rectangle((b['x']-b['w']/2)*self.w,(b['y']-b['h']/2)*self.h,
                (b['x']+b['w']/2)*self.w,(b['y']+b['h']/2)*self.h,outline='#00ff88',width=3,tags='box')

    def begin(self,event):
        self.start=(event.x,event.y)

    def drag(self,event):
        if self.start:
            self.canvas.delete('box')
            self.canvas.create_rectangle(*self.start,max(0,min(self.w,event.x)),max(0,min(self.h,event.y)),
                                         outline='#00ff88',width=3,tags='box')

    def end(self,event):
        if self.start:
            try:
                self.boxes=[box_from_drag(self.start,(event.x,event.y),self.w,self.h,self.row['food_name'])]
            except ValueError as exc:
                messagebox.showwarning('Chon lai',str(exc),parent=self)
            self.start=None
            self.draw_saved()

    def save(self):
        try:
            save_boxes(self.parent.dataset,self.row['image_id'],self.boxes,self.reviewer.get())
            self.parent.reload()
            self.destroy()
        except (ValueError,OSError) as exc:
            messagebox.showerror('Chua luu box',str(exc),parent=self)

    def clear(self):
        save_boxes(self.parent.dataset,self.row['image_id'],[],self.reviewer.get(),status='Pending')
        self.boxes=[]
        self.draw_saved()
        self.parent.reload()


class SourceWindow(tk.Toplevel):
    def __init__(self,parent,row):
        super().__init__(parent)
        self.parent,self.row=parent,row
        self.title('Kiem tra nguon anh')
        self.geometry('800x450')
        source=parent.dataset.sources()[row['source_dataset']]
        ttk.Label(self,text='Google chi la cong cu tim kiem. Chi duyet quyen khi da kiem tra trang nguon / giay phep.',wraplength=760).pack(padx=12,pady=12)
        form=ttk.Frame(self,padding=12)
        form.pack(fill='both',expand=True)
        self.vars={}
        fields=[('status','Trang thai',source.get('license_status','Pending')),('license_id','Ten / ID giay phep',source.get('license_id','')),
                ('license_url','URL giay phep',source.get('license_url','')),('source_page','Trang nguon anh',source.get('source_page','')),
                ('reviewer','Nguoi kiem tra',source.get('license_reviewer','')),('evidence','Can cu quyen su dung',source.get('license_evidence',''))]
        for i,(key,label,value) in enumerate(fields):
            ttk.Label(form,text=label).grid(row=i,column=0,sticky='w',pady=6)
            var=tk.StringVar(value=value)
            self.vars[key]=var
            widget=ttk.Combobox(form,textvariable=var,values=['Pending','Approved','Rejected'],state='readonly') if key=='status' else ttk.Entry(form,textvariable=var)
            widget.grid(row=i,column=1,sticky='ew',padx=8)
        form.columnconfigure(1,weight=1)
        ttk.Button(self,text='Luu kiem tra nguon',command=self.save).pack(pady=12)

    def save(self):
        try:
            save_source_review(self.parent.dataset,self.row['source_dataset'],**{k:v.get() for k,v in self.vars.items()})
            self.parent.reload()
            self.destroy()
        except (ValueError,OSError) as exc:
            messagebox.showerror('Chua luu',str(exc),parent=self)

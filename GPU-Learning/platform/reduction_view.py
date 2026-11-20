"""Resizable, lifecycle-safe Tk reduction teaching workspace."""
import tkinter as tk
from tkinter import ttk
from reduction_model import trace,two_stage

SCENARIOS={'A · 8 Threads / Shared':('shared',8),'B · 32 Lanes / Shuffle':('shuffle',32),'B · 8 values + 24 zeros':('shuffle',8),'C · 256 Threads / 8 Warps':('block',256),'D · 513 values / Two kernels':('two',513)}
class ReductionView(ttk.Frame):
    def __init__(self,parent):
        super().__init__(parent);self.pack(fill='both',expand=True);self.timer=None;self.index=0
        self.mode=tk.StringVar(value=next(iter(SCENARIOS)));self.op=tk.StringVar(value='SUM');self.speed=tk.DoubleVar(value=1)
        ttk.Label(self,text='教学模拟 / CPU 状态计算 · 不是 GPU 实时执行轨迹',foreground='#61d6bd').pack(anchor='w')
        bar=ttk.Frame(self);bar.pack(fill='x',pady=8)
        for variable,values,width in [(self.mode,list(SCENARIOS),34),(self.op,['SUM','MAX','MIN'],7)]:
            combo=ttk.Combobox(bar,textvariable=variable,values=values,state='readonly',width=width);combo.pack(side='left',padx=3);combo.bind('<<ComboboxSelected>>',lambda _:self.load())
        controls=ttk.Frame(self);controls.pack(fill='x')
        for title,command in [('播放',self.play),('暂停',self.pause),('上一步',lambda:self.move(-1)),('单步',lambda:self.move(1)),('重置',self.reset)]:ttk.Button(controls,text=title,command=command).pack(side='left',padx=2)
        ttk.Label(controls,text='速度').pack(side='left',padx=6);ttk.Scale(controls,from_=.25,to=3,variable=self.speed,length=100).pack(side='left')
        self.info=ttk.Label(self,wraplength=850);self.info.pack(anchor='w',pady=10)
        ttk.Label(self,text='绿色：写入/参与合并   蓝色边框：数据来源   灰色：本阶段不更新；点击线程查看算式',foreground='#9baec6').pack(anchor='w')
        tabs=ttk.Notebook(self);tabs.pack(fill='both',expand=True,pady=8)
        self.canvases=[]
        for name in ['Thread / Warp / Grid','Shared Memory 读写']:
            frame=ttk.Frame(tabs);tabs.add(frame,text=name)
            canvas=tk.Canvas(frame,bg='#0c1119',highlightthickness=0);scroll=ttk.Scrollbar(frame,command=canvas.yview);canvas.configure(yscrollcommand=scroll.set)
            scroll.pack(side='right',fill='y');canvas.pack(fill='both',expand=True);canvas.bind('<Configure>',lambda _:self.draw());self.canvases.append(canvas)
        self.detail=ttk.Label(self,wraplength=880);self.detail.pack(anchor='w')
        self.load()
    def load(self):
        self.pause();kind,n=SCENARIOS[self.mode.get()];values=list(range(1,n+1))
        self.states=two_stage(values,self.op.get()) if kind=='two' else trace(values,kind,self.op.get());self.index=0;self.draw()
    def pause(self):
        if self.timer is not None:self.after_cancel(self.timer);self.timer=None
    def move(self,amount):self.pause();self.index=max(0,min(len(self.states)-1,self.index+amount));self.draw()
    def reset(self):self.move(-len(self.states))
    def play(self):
        if self.timer is None and self.index<len(self.states)-1:self.timer=self.after(int(900/self.speed.get()),self.tick)
    def tick(self):self.timer=None;self.index=min(self.index+1,len(self.states)-1);self.draw();self.play()
    def destroy(self):self.pause();super().destroy()
    def inspect(self,i):
        s=self.states[self.index];sources=dict(s.edges);j=sources.get(i)
        self.detail.configure(text=f'Thread {i} / Warp {i//32} / Lane {i%32}: '+(f'{s.before[i]} {self.op.get()} Lane {j%32} ({s.before[j]}) → {s.after[i]}' if j is not None else f'value={s.after[i]}；本阶段无合并'))
    def draw(self):
        if not hasattr(self,'states'):return
        s=self.states[self.index];active=dict(s.edges);sources=set(active.values())
        self.info.configure(text=f'阶段 {self.index}/{len(self.states)-1} · Block {s.block} · Offset {s.offset or "—"} · Mask {s.mask}\n{s.title}\n{s.boundary}')
        for idx,canvas in enumerate(self.canvases):
            canvas.delete('all');values=s.after if idx==0 else s.shared
            width=max(500,canvas.winfo_width());cols=max(4,min(16,int(width//95)));cell=width/cols;y=10
            if not values:canvas.create_text(15,25,text='当前阶段使用寄存器 / global partial，无 Shared Memory 操作',fill='#9baec6',anchor='w')
            for i,value in enumerate(values):
                if i%32==0:
                    if i:y+=12
                    canvas.create_text(12,y+8,text=f'{"Warp" if idx==0 else "Shared group"} {i//32}',fill='#9baec6',anchor='w');y+=28
                c=i%32%cols;r=(i%32)//cols;x=c*cell+6;cy=y+r*70
                fill='#244a43' if i in active else '#192231';outline='#70aeff' if i in sources else '#354254'
                canvas.create_rectangle(x,cy,x+cell-12,cy+59,fill=fill,outline=outline,width=2)
                label=f'T{i} / L{i%32}' if idx==0 else f'smem[{i}]'
                old=s.before[i] if i<len(s.before) else value
                text=f'{label}\n{old} → {value}' if i in active else f'{label}\n{value}'
                item=canvas.create_text(x+8,cy+28,text=text,fill='#e4ecf7',anchor='w',font=('Consolas',9))
                if idx==0:canvas.tag_bind(item,'<Button-1>',lambda _,j=i:self.inspect(j))
                if i%32==31 or i==len(values)-1:y+=((i%32)//cols+1)*70
            canvas.configure(scrollregion=(0,0,width,y+30))
        self.detail.configure(text='Shuffle: 全部 32 Lane 执行，mask=0xffffffff；越界来源不参与合并。填充 Lane 是有效执行线程，持有操作单位元。' if s.mask!='—' else 'Shared: 绿色线程读取自身与 offset 来源位置，写回自身位置；同步单独作为一个阶段。')

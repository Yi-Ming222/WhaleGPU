"""GPU Learning Workbench: standard-library Windows desktop application."""
import json
import queue
import statistics
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from runner import ROOT, DATA, EXPERIMENTS, execute
from lessons import LESSONS, reduction_steps

BG, PANEL, FG, MUTED, ACCENT = '#10151e', '#192231', '#e4ecf7', '#9baec6', '#61d6bd'

class Workbench(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('GPU Learning • CUDA Workbench')
        self.geometry('1280x820'); self.minsize(1000, 660); self.configure(bg=BG)
        self.protocol('WM_DELETE_WINDOW', self.close_requested)
        self.events = queue.Queue(); self.busy = False; self.last = None; self.playing = False
        self.progress_file = ROOT / 'records' / 'learning.json'
        try: self.completed = set(json.loads(self.progress_file.read_text(encoding='utf-8')))
        except (OSError, ValueError): self.completed = set()
        style = ttk.Style(self); style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG, font=('Segoe UI', 11))
        style.configure('TButton', background=PANEL, foreground=FG, padding=10)
        style.map('TButton', background=[('active', '#30445d')])
        nav = tk.Frame(self, bg=PANEL, width=220); nav.pack(side='left', fill='y'); nav.pack_propagate(False)
        tk.Label(nav, text='GPU / LEARNING', bg=PANEL, fg=ACCENT, font=('Segoe UI', 17, 'bold'), pady=28).pack()
        for name in ['Learning Center', 'Kernel Laboratory', 'GPU Visualization', 'Benchmark', 'Performance Analysis']:
            ttk.Button(nav, text=name, command=lambda n=name:self.show(n)).pack(fill='x', padx=12, pady=5)
        tk.Label(nav, text='CUDA 13.0 · sm_120\nNative CUDA / Python UI', bg=PANEL, fg=MUTED, pady=30).pack(side='bottom')
        self.main = ttk.Frame(self, padding=24); self.main.pack(side='left', fill='both', expand=True)
        self.status = tk.StringVar(value='Ready • 独立 CUDA 实验可直接运行')
        self.show('Learning Center'); self.after(100, self.poll)

    def close_requested(self):
        if self.busy:
            self.status.set('任务执行中；请等待完成或超时后关闭窗口。')
        else: self.destroy()

    def text(self, parent, content='', height=15):
        frame=ttk.Frame(parent); frame.pack(fill='both', expand=True, pady=8)
        box=tk.Text(frame, bg='#0c1119', fg=FG, insertbackground=FG, font=('Consolas', 11), wrap='word', height=height, padx=14, pady=12)
        scroll=ttk.Scrollbar(frame, command=box.yview); box.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); box.pack(fill='both', expand=True)
        box.insert('1.0',content); box.configure(state='disabled'); return box

    def show(self, name):
        self.page=name; self.playing=False
        for child in self.main.winfo_children(): child.destroy()
        ttk.Label(self.main,text=name,font=('Segoe UI',24,'bold')).pack(anchor='w')
        ttk.Label(self.main,textvariable=self.status,foreground=MUTED).pack(anchor='w',pady=(5,18))
        if name=='Learning Center': self.learning()
        elif name=='Kernel Laboratory': self.lab()
        elif name=='Benchmark': self.benchmark()
        elif name=='GPU Visualization': self.visualization()
        else: self.performance()

    def learning(self):
        row=ttk.Frame(self.main); row.pack(fill='x')
        self.lesson=tk.StringVar(value=next(iter(LESSONS)))
        menu=ttk.Combobox(row,textvariable=self.lesson,values=list(LESSONS),state='readonly',width=32)
        menu.pack(side='left'); menu.bind('<<ComboboxSelected>>',lambda _:self.lesson_content())
        ttk.Button(row,text='标记已学',command=self.complete).pack(side='left',padx=12)
        self.progress=ttk.Label(self.main,text=f'已完成 {len(self.completed)} / {len(LESSONS)}'); self.progress.pack(anchor='w',pady=12)
        self.lesson_box=self.text(self.main); self.lesson_content()

    def replace(self,box,text):
        box.configure(state='normal'); box.delete('1.0','end'); box.insert('1.0',text); box.configure(state='disabled')

    def lesson_content(self): self.replace(self.lesson_box,LESSONS[self.lesson.get()])

    def complete(self):
        self.completed.add(self.lesson.get()); self.progress_file.parent.mkdir(exist_ok=True)
        self.progress_file.write_text(json.dumps(sorted(self.completed),ensure_ascii=False),encoding='utf-8')
        self.progress.configure(text=f'已完成 {len(self.completed)} / {len(LESSONS)}')

    def lab(self):
        row=ttk.Frame(self.main); row.pack(fill='x')
        ttk.Label(row,text='01 / Vector Addition • 1,048,579 float elements').pack(side='left')
        ttk.Button(row,text='编译',command=lambda:self.start('build')).pack(side='right',padx=4)
        ttk.Button(row,text='编译并运行',command=lambda:self.start('run')).pack(side='right')
        notebook=ttk.Notebook(self.main); notebook.pack(fill='both',expand=True,pady=12)
        source=ttk.Frame(notebook); output=ttk.Frame(notebook)
        notebook.add(source,text='main.cu • 原生源码'); notebook.add(output,text='实际执行记录')
        self.text(source,EXPERIMENTS['vector-add'].read_text(encoding='utf-8'))
        self.text(output,self.report(self.last) if self.last else '尚未执行。编译和运行结束后在此查看真实输出。')
        ttk.Button(self.main,text='读取最近保存记录',command=self.history).pack(anchor='w')

    def start(self,action):
        if self.busy: self.status.set('已有任务执行中，请等待完成'); return
        try: repeats=int(self.repeats.get()) if action=='benchmark' else 1
        except ValueError: self.status.set('运行次数必须为 1–20 的整数'); return
        if not 1<=repeats<=20: self.status.set('运行次数必须为 1–20'); return
        self.busy=True; self.status.set('Running • 后台编译 / GPU 执行中，可切换页面')
        def work():
            try: self.events.put(execute(action,repeats=repeats))
            except Exception as e: self.events.put({'passed':False,'error':str(e),'samples':[]})
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            self.last=self.events.get_nowait(); self.busy=False
            self.status.set(('PASS' if self.last['passed'] else 'FAIL')+' • '+self.last.get('error','真实执行完成，记录已保存'))
            self.show(self.page)
        except queue.Empty: pass
        self.after(100,self.poll)

    def report(self,r):
        if not r:return '尚无实验结果。'
        chunks=[f"Status: {'PASS' if r.get('passed') else 'FAIL'}",r.get('error','')]
        if 'compile' in r: chunks += ['--- nvcc ---',r['compile']['stdout'],r['compile']['stderr']]
        for i,s in enumerate(r.get('samples',[])):
            chunks += [f"--- Run {i+1}: {s['process_ms']:.3f} ms / exit {s['exit_code']} ---",s['stdout'],s['stderr']]
        return '\n'.join(chunks)

    def history(self):
        records=sorted((p for p in DATA.glob('*.json') if p.name!='learning.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        if records:
            try: self.last=json.loads(records[0].read_text(encoding='utf-8')); self.show(self.page)
            except (OSError,ValueError) as e:self.status.set(str(e))
        else:self.status.set('尚无保存记录')

    def benchmark(self):
        ttk.Label(self.main,text='Vector Addition / 固定输入规模 1,048,579 / 每次独立进程').pack(anchor='w')
        row=ttk.Frame(self.main);row.pack(fill='x',pady=12)
        self.repeats=tk.StringVar(value='5'); ttk.Spinbox(row,from_=1,to=20,textvariable=self.repeats,width=5).pack(side='left')
        ttk.Button(row,text='编译并重复测试',command=lambda:self.start('benchmark')).pack(side='left',padx=12)
        ttk.Button(row,text='最近保存记录',command=self.history).pack(side='left')
        ttk.Label(self.main,text='计时：进程启动 → CUDA 初始化 / 分配 / 传输 / Kernel / CPU 校验 → 退出\n编译时间不计入；无预热，包含冷启动影响。Kernel-only 时间尚未采集。',foreground=MUTED).pack(anchor='w')
        samples=(self.last or {}).get('samples',[])
        if samples:
            times=[s['process_ms'] for s in samples]
            ttk.Label(self.main,text=f'Mean {statistics.mean(times):.3f} ms     Min {min(times):.3f}     Max {max(times):.3f}',font=('Consolas',16)).pack(anchor='w',pady=18)
            canvas=tk.Canvas(self.main,height=150,bg=PANEL,highlightthickness=0);canvas.pack(fill='x')
            for i,t in enumerate(times):
                x=15+i*36;h=t/max(times)*110
                canvas.create_rectangle(x,125-h,x+25,125,fill=ACCENT,outline='')
                canvas.create_text(x+12,140,text=str(i+1),fill=FG)
        self.text(self.main,self.report(self.last),height=8)

    def performance(self):
        self.text(self.main,'PERFORMANCE DATA CONTRACT\n\nKernel time: 未采集\nGPU utilization: 未采集\nMemory throughput: 未采集\nOccupancy: 未采集\nWarp metrics: 未采集\nKernel timeline: 未采集\n\n当前只有 CPU 侧进程墙钟计时与 CUDA 程序正确性输出。\n不能从这些数据推断 GPU Occupancy 或硬件瓶颈。\n\n后续接口：导入 Nsight Compute / Nsight Systems 的导出记录，保留采集工具版本、指标单位和实验 ID。\n\n版本与输入规模对比将在增加原生实验参数协议后接入。')

    def visualization(self):
        from reduction_view import ReductionView
        self.reduction_view = ReductionView(self.main)

if __name__=='__main__': Workbench().mainloop()



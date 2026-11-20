"""Generate deterministic CPU teaching traces for the browser."""
from trace import reduction_rounds, thread_indices


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}：请输入 {low} 到 {high} 之间的整数。")
    return value


def frame(title, detail, **state):
    return dict(title=title, detail=detail, **state)


def dispatch(xs, blocks, threads, indices, step):
    frames = [frame("启动 Grid", f"创建 {blocks} 个 Block，每块 {threads} 个线程。")]
    rounds = max(map(len, indices), default=0)
    for turn in range(rounds):
        reads = [[tid, owned[turn]] for tid, owned in enumerate(indices) if turn < len(owned)]
        frames.append(frame(f"第 {turn + 1} 轮访问",
                            f"{len(reads)} 个线程读入数据。下一轮的下标增加 {step}。",
                            links=reads))
    counts = [0] * len(xs)
    for owned in indices:
        for i in owned:
            counts[i] += 1
    repeated = sum(count > 1 for count in counts)
    frames.append(frame("索引访问完成", f"{len(xs)} 个元素中，{repeated} 个被多个线程访问。",
                        coverage=counts))
    return frames


def reduce_trace(xs, blocks, threads, indices, pairing):
    frames = [frame("启动第一阶段 Kernel", "每个线程从自己的全局编号开始，以 Grid 总线程数为步长读入数据。",
                    stage=1)]
    totals = [[0] * threads for _ in range(blocks)]
    for turn in range(max(map(len, indices), default=0)):
        reads = []
        for tid, owned in enumerate(indices):
            if turn < len(owned):
                i = owned[turn]
                totals[tid // threads][tid % threads] += xs[i]
                reads.append([tid, i])
        frames.append(frame(f"第 {turn + 1} 轮读取", "各线程先在自己的局部累加器中求和。",
                            stage=1, links=reads, local=[row.copy() for row in totals]))

    frames.append(frame("写入共享内存", "每个线程把局部小计写到本 Block 的共享内存，空闲线程贡献 0。",
                        stage=1, shared=totals))
    frames.append(frame("块内同步", "各 Block 分别等待自己的所有线程完成写入。",
                        stage=1, shared=totals, barrier=True))
    rounds = [list(reduction_rounds(row, pairing)) for row in totals]
    shared = totals
    for turn, group in enumerate(zip(*rounds), 1):
        shared = [row for _, row, _ in group]
        ops = [ops for _, _, ops in group]
        frames.append(frame(f"块内归约 · 第 {turn} 轮", "高亮线程读取两个槽位，将和写回自己的槽位。",
                            stage=1, shared=shared, ops=ops))
        frames.append(frame("块内同步", "下一轮读取前，块内各线程等待这一轮写入完成。",
                            stage=1, shared=shared, barrier=True))
    partials = [row[0] for row in shared]
    frames.append(frame("第一阶段完成", "每个 Block 的线程 0 将块小计写到全局内存。",
                        stage=1, shared=shared, partials=partials))

    width = 1 << (blocks - 1).bit_length()
    shared = partials + [0] * (width - blocks)
    frames.append(frame("启动第二阶段 Kernel", "第一阶段全部结束后，启动一个新 Block 读取块小计，缺位用 0 补齐。",
                        stage=2, shared=[shared], partials=partials))
    frames.append(frame("第二阶段块内同步", "同一个新 Block 中的线程等待共享内存写入完成。",
                        stage=2, shared=[shared], partials=partials, barrier=True))
    for turn, (_, shared, ops) in enumerate(reduction_rounds(shared, pairing), 1):
        frames.append(frame(f"最终归约 · 第 {turn} 轮", "第二阶段的线程合并第一阶段产生的块小计。",
                            stage=2, shared=[shared], partials=partials, ops=[ops]))
        frames.append(frame("第二阶段块内同步", "所有参与线程到达屏障后才读取最后结果。",
                            stage=2, shared=[shared], partials=partials, barrier=True))
    frames.append(frame("归约完成", f"最终结果 {shared[0]}，CPU 参考和 {sum(xs)}。",
                        stage=2, shared=[shared], partials=partials, result=shared[0]))
    return frames



def shared_trace(xs, blocks, threads):
    shared = [[None] * threads for _ in range(blocks)]
    arrivals = [[] for _ in range(blocks)]
    neighbors = [[None] * threads for _ in range(blocks)]
    frames = [frame("分配块内共享内存", "每个 Block 有独立的槽位和屏障。每个线程向自己的槽位写入数据。", shared=[row.copy() for row in shared])]
    for tick in range(threads + 1 + (blocks - 1) * 2):
        writes, reads = [], []
        activity = [[] for _ in range(blocks)]
        for block in range(blocks):
            lane = tick - block * 2
            if 0 <= lane < threads:
                i = block * threads + lane
                shared[block][lane] = xs[i]
                arrivals[block].append(lane)
                activity[block].append(lane)
                writes.append([i, i])
            elif lane == threads:
                row = shared[block]
                neighbors[block] = [value + (row[i + 1] if i + 1 < threads else 0)
                                    for i, value in enumerate(row)]
                activity[block] = list(range(threads))
                reads.append(block)
        released = [len(lanes) == threads for lanes in arrivals]
        waiting = [block for block, lanes in enumerate(arrivals) if lanes and not released[block]]
        title = "写入并抵达屏障" if not reads else "已放行的 Block 读取共享内存"
        parts = [f"Block {b} 读取自己的槽位和右邻居，块尾补 0。" for b in reads]
        parts += [f"Block {b} 已有 {len(arrivals[b])}/{threads} 个线程抵达，仍需等待。" for b in waiting]
        parts += [f"Block {b} 到齐，屏障已放行。" for b in range(blocks)
                  if tick == b * 2 + threads - 1]
        frames.append(frame(title, " ".join(parts), links=writes,
                            shared=[row.copy() for row in shared], activity=activity,
                            neighbors=[row.copy() for row in neighbors],
                            sync=[dict(arrived=lanes.copy(), released=done)
                                  for lanes, done in zip(arrivals, released)]))

    return frames


def race_trace(threads, schedule):
    values = list(range(1, threads + 1))
    cache = [None] * threads
    pending = [None] * threads
    total = atomic = 0
    expected = sum(values)
    frames = [frame("同一个全局地址", "每个线程尝试向 total 加上自己的值。两边分别从 0 开始。",
                    total=0, atomic=0, cache=cache.copy(), pending=pending.copy())]
    phases = ("read", "add", "write")
    steps = ([(lane, phase) for phase in phases for lane in range(threads)]
             if schedule == "overlap" else
             [(lane, phase) for lane in range(threads) for phase in phases])
    for lane, phase in steps:
        value = values[lane]
        before = atomic
        if phase == "read":
            cache[lane] = total
            title, detail = "读取 total", f"T{lane} 读到 {total}，保存在自己的寄存器里。"
        elif phase == "add":
            pending[lane] = cache[lane] + value
            title = "计算新值"
            detail = f"T{lane} 计算 {cache[lane]} + {value} = {pending[lane]}，还未更新共享地址。"
        else:
            total = pending[lane]
            atomic += value
            title = "写回 / 原子加"
            detail = (f"左侧 T{lane} 写回旧快照计算的 {total}。"
                      f"右侧 T{lane} 对当前值做一次原子加：{before} + {value} = {atomic}。")
        frames.append(frame(title, detail, total=total, atomic=atomic, lane=lane,
                            cache=cache.copy(), pending=pending.copy(), phase=phase))
    frames.append(frame("比较最终结果",
                        f"参考和 {expected}。非原子结果 {total}，原子结果 {atomic}。"
                        "顺序是教学中选定的一种调度，真实 GPU 可能出现其他交错。",
                        total=total, atomic=atomic, cache=cache, pending=pending,
                        lost=expected - total, expected=expected))
    return dict(source="cpu-simulation", mode="race", threads=threads, values=values, frames=frames)



def run_experiment(p):
    mode = p.get("mode", "reduce")
    if mode not in ("threads", "stride", "reduce", "shared", "race"):
        raise ValueError("未知实验。")
    if mode == "race":
        count = integer(p.get("threads", 2), 1, 64, "线程数")
        schedule = p.get("schedule", "overlap")
        if schedule not in ("overlap", "serial"):
            raise ValueError("未知模拟调度。")
        return race_trace(count, schedule)
    blocks = integer(p.get("blocks", 2), 1, 8, "Block 数")
    threads = integer(p.get("threads", 4), 1, 64, "每块线程数")
    if mode == "reduce" and threads & (threads - 1):
        raise ValueError("归约的每块线程数须为 2 的幂。")
    if mode in ("threads", "stride"):
        length = integer(p.get("length", 13), 0, 1024, "数组长度")
        xs = list(range(1, length + 1))
    elif mode == "shared":
        xs = list(range(1, blocks * threads + 1))
    else:
        xs = p.get("values", list(range(1, 14)))
    if not isinstance(xs, list) or len(xs) > 1024:
        raise ValueError("请输入最多 1024 个整数。")
    for x in xs:
        integer(x, -1000000, 1000000, "输入值")
    pairing = p.get("pairing", "halves")
    if pairing not in ("halves", "adjacent"):
        raise ValueError("未知归约配对方式。")
    step = 1 if mode == "stride" and p.get("unit") is True else blocks * threads
    indices = thread_indices(len(xs), blocks, threads, step)
    if mode == "reduce":
        frames = reduce_trace(xs, blocks, threads, indices, pairing)
    elif mode == "shared":
        frames = shared_trace(xs, blocks, threads)
    else:
        frames = dispatch(xs, blocks, threads, indices, step)
    return dict(source="cpu-simulation", mode=mode, blocks=blocks, threads=threads,
                step=step, values=xs, indices=indices, frames=frames)




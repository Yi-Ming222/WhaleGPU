
'use strict';

const $ = id => document.getElementById(id);
const palettes = ['#66e6f5', '#86aaff', '#c5a0ff', '#69e0b6', '#f2c47f', '#f5a4bf', '#a7db84', '#7ad6fc'];
const modes = {
  reduce: ['归约实验', 'Parallel Reduction', '从 13 个元素出发，观察线程小计如何合并为块小计，再由第二阶段 Kernel 汇总。'],
  threads: ['线程拓扑', 'Grid → Block → Thread', '展开 Grid 的线程布局。点击一个 Block 或线程，查看它负责的数组元素。'],
  stride: ['网格步长循环', 'Grid-Stride Loop', '对比相邻递增与跨整个 Grid 跳跃，观察访问覆盖和重叠。'],
  shared: ['共享内存与同步', 'Shared Memory / Barrier', '观察线程写入、抵达屏障和读取邻居的过程。每个 Block 独立释放。'],
  race: ['数据竞争', 'Race Condition / Atomic Add', '在相同线程数下，对比非原子读改写与原子加。你可以切换模拟调度。'],
};
let mode = 'reduce';
let experiment, cursor = 0, timer = null, request = 0, loading = false;
let advanced = false, selection = {block: 0, stage: 1};
let lastScroll = 0;

function setup() {
  $('nav').innerHTML = Object.entries(modes).map(([key, info], i) =>
    `<button data-mode="${key}" aria-current="${key === mode}"><span>0${i + 1}</span>${info[0]}</button>`).join('');
  $('block-label').hidden = mode === 'race';
  $('length-label').hidden = !['threads', 'stride'].includes(mode);
  $('value-label').hidden = mode !== 'reduce';
  $('pairing-label').hidden = mode !== 'reduce';
  $('step-label').hidden = mode !== 'stride';
  $('schedule-label').hidden = mode !== 'race';
  $('blocks').value = mode === 'stride' ? 1 : 2;
  $('threads').value = mode === 'race' ? 2 : 4;
  $('length').value = mode === 'stride' ? 16 : 13;
  $('values').value = Array.from({length:13}, (_, i) => i + 1).join(', ');
  $('pairing').value = 'halves';
  $('step').value = 'grid';
  $('schedule').value = 'overlap';
  selection = {block: 0, stage: 1};
  lastScroll = 0;
  load(true);
}

async function post(path, payload) {
  const response = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '实验请求失败。');
  return data;
}

async function load(clear = false) {
  stop();
  const id = ++request;
  loading = true;
  if (clear) experiment = null;
  $('error').textContent = '';
  render();
  try {
    const p = {mode, blocks: Number($('blocks').value), threads: Number($('threads').value)};
    if (mode === 'reduce') {
      const parts = $('values').value.trim().split(/[\s,，]+/).filter(Boolean);
      if (parts.some(s => !/^[+-]?\d+$/.test(s))) throw new Error('输入数组只接受整数。');
      p.values = parts.map(Number);
      p.pairing = $('pairing').value;
    } else if (mode === 'stride' || mode === 'threads') {
      p.length = Number($('length').value);
      p.unit = $('step').value === 'unit';
    } else if (mode === 'race') {
      p.schedule = $('schedule').value;
    }
    const next = await post('/api/experiment', p);
    if (id !== request) return;
    experiment = next;
    cursor = 0;
    selection = {block: 0, stage: 1};
    lastScroll = 0;
  } catch (error) {
    if (id === request) $('error').textContent = error.message;
  } finally {
    if (id === request) {
      loading = false;
      render();
    }
  }
}

function stop() {
  clearInterval(timer);
  timer = null;
}

function seek(value) {
  if (!experiment || loading) return;
  cursor = Math.max(0, Math.min(experiment.frames.length - 1, value));
  if (cursor === experiment.frames.length - 1) stop();
  lastScroll = $('canvas').scrollTop;
  render();
}

function togglePlay() {
  if (!experiment || loading) return;
  if (timer) {
    stop();
    renderControls();
    return;
  }
  if (cursor === experiment.frames.length - 1) seek(0);
  timer = setInterval(() => seek(cursor + 1), Number($('speed').value));
  renderControls();
}

function renderControls() {
  const total = experiment?.frames.length || 0;
  $('play').textContent = timer ? 'Ⅱ 暂停' : '▶ 自动播放';
  $('play').disabled = loading || total < 2;
  $('back').disabled = loading || cursor < 1;
  $('next').disabled = loading || cursor >= total - 1;
  $('timeline').disabled = loading || total < 2;
  $('timeline').max = Math.max(0, total - 1);
  $('timeline').value = cursor;
  $('position').textContent = total ? `STEP ${cursor + 1} / ${total}` : 'STEP —';
  $('counter').textContent = total ? `${cursor + 1} / ${total}` : '—';
}

function render() {
  const [title, english, intro] = modes[mode];
  const run = experiment;
  const f = run?.frames[cursor];
  const grid = run && mode !== 'race'
    ? `<span class="chip" title="Grid 是一次 Kernel 启动包含的全部 Block">${run.blocks} Block × ${run.threads} Thread</span>`
    : '';
  $('hero').innerHTML = `<div class="eyebrow">GPU LAB / ${english}</div><h1>${title}</h1><p>${intro}</p>
    <div class="hero-tags"><span class="chip" title="Python 生成用于教学的执行轨迹">● CPU 教学模拟</span>${grid}
    ${run && ['stride','threads'].includes(mode) ? `<span class="chip" title="stride 等于整个 Grid 的线程数">索引增量 ${run.step}</span>` : ''}</div>`;
  renderControls();
  if (!f) {
    $('stage').textContent = '等待轨迹';
    $('canvas').textContent = loading ? '正在生成轨迹…' : '请检查参数后重新运行。';
    $('detail').textContent = '';
    $('selection').textContent = '';
    return;
  }
  const stage = f.stage || 1;
  $('stage').textContent = mode === 'reduce' ? `KERNEL 0${stage}` : 'CPU SIM';
  const activeSelection = selection.stage === stage ? selection : {block:0, stage};
  $('canvas').innerHTML = mode === 'race' ? raceCanvas(f) : gridCanvas(run, f, activeSelection);
  $('canvas').scrollTop = lastScroll;
  drawLinks(f);
  $('detail').innerHTML = `<h3>${f.title}</h3><p>${f.detail}</p>
    ${advanced && mode !== 'race' ? `<p><code>${mode === 'shared' ? '__syncthreads() · block scope' : 'tid = blockIdx.x × blockDim.x + threadIdx.x'}</code></p>` : ''}`;
  $('selection').innerHTML = mode === 'race' ? raceInspector(f) : inspector(run, f, activeSelection);
}

function gridCanvas(run, f, selected) {
  const second = f.stage === 2;
  const used = new Map((f.links || []).map(([tid, index]) => [tid, index]));
  const marks = new Map((f.links || []).map(([tid, index]) => [index, palettes[Math.floor(tid / run.threads)]]));
  const values = f.partials || run.values;
  const data = `<div class="region"><div class="region-title">${f.partials ? '第一阶段输出 / BLOCK PARTIALS' : '全局内存 / INPUT'} · ${values.length}</div>
    <div class="cells data">${values.map((value, index) => {
      const color = marks.get(index);
      const counts = f.coverage?.[index];
      return `<span id="data-${index}" class="slot ${color ? 'active' : ''} ${counts > 1 ? 'risk' : ''}" ${color ? `style="--accent:${color}"` : ''}
        title="下标 ${index}${counts == null ? '' : ` · 访问 ${counts} 次`}"><small>${f.partials ? 'B' : ''}${index}</small>${value}</span>`;
    }).join('') || '<span class="subtle">空数组</span>'}</div></div>`;
  const blocks = second ? 1 : run.blocks;
  const threads = second ? f.shared[0].length : run.threads;
  const cards = Array.from({length:blocks}, (_, block) => {
    const ops = f.ops?.[block] || [];
    const sync = f.sync?.[block];
    const current = ops.map(o => o.lane);
    const partner = ops.map(o => o.partner);
    const color = palettes[block];
    const lanes = Array.from({length:threads}, (_, lane) => {
      const tid = block * threads + lane;
      const active = used.has(tid) || current.includes(lane) || f.activity?.[block]?.includes(lane);
      const showId = advanced ? `G${tid}` : `T${lane}`;
      return `<button id="lane-${tid}" type="button" class="lane ${active ? 'active' : ''}" data-block="${block}" data-lane="${lane}"
        aria-pressed="${selected.block === block && selected.lane === lane}" title="Block ${block} · Thread ${lane} · 全局编号 ${tid}">${showId}<small>${advanced ? `T${lane}` : `#${tid}`}</small></button>`;
    }).join('');
    const slots = f.shared?.[block] || f.local?.[block];
    const memory = slots ? `<div class="shared"><div class="region-title">${f.shared ? '共享内存' : '线程局部累加器'}</div><div class="cells">${slots.map((value, lane) =>
      `<span class="slot ${current.includes(lane) ? 'active' : ''} ${partner.includes(lane) ? 'partner' : ''} ${sync?.arrived.includes(lane) && !sync.released ? 'wait' : ''}" title="槽位 ${lane}"><small>${lane}</small>${value ?? '·'}</span>`).join('')}</div></div>` : '';
    const equations = ops.length ? `<div class="equations" aria-label="本轮归约树节点">${ops.map(o => `<span class="equation" title="T${o.lane} 合并槽位 ${o.lane} 和 ${o.partner}">T${o.lane} · ${o.left} + ${o.right} → ${o.result}</span>`).join('')}</div>` : '';
    const neighbors = f.neighbors?.[block];
    const readout = neighbors?.some(value => value != null) ? '<div class="shared"><div class="region-title">读取并求和</div><div class="cells">' + neighbors.map((value, lane) => '<span class="slot active" title="线程 ' + lane + ' 的邻居和"><small>T' + lane + '</small>' + value + '</span>').join('') + '</div></div>' : '';
    const barrier = sync ? `<div class="barrier">屏障 ${sync.arrived.length}/${threads} · ${sync.released ? '已放行' : '等待'}</div>`
      : f.barrier ? `<div class="barrier">屏障 · ${threads}/${threads} 到齐</div>` : '';
    return `<div class="block ${selected.block === block ? 'selected' : ''} ${sync?.released ? 'released' : ''}" style="--accent:${color}">
      <button class="block-title" data-block="${block}" title="查看这个 Block 内的所有线程">▦ Block ${block}</button>
      <div class="lanes">${lanes}</div>${memory}${readout}${equations}${barrier}</div>`;
  }).join('');
  return `${data}<div class="region"><div class="region-title">${second ? '第二阶段 / FINAL GRID' : '线程网格 / GRID'} · ${blocks} × ${threads}</div><div class="blocks">${cards}</div></div>
    ${f.result == null ? '' : `<div class="region result">最终结果 · ${f.result}</div>`}
    ${reductionTree(run, f, selected)}<svg class="links" aria-hidden="true"></svg>`;
}



function reductionTree(run, f, selected) {
  if (!f.shared || mode !== 'reduce') return '';
  const block = selected.block;
  const frames = run.frames.slice(0, cursor + 1).filter(step => step.stage === f.stage);
  const start = frames.find(step => step.shared);
  const leaves = start.shared[block].map((value, lane) => ({value, lane, depth: 0}));
  const nodes = [...leaves];
  const roots = [...leaves];
  const edges = [];
  let depth = 0;
  for (const step of frames) {
    if (!step.ops) continue;
    depth += 1;
    for (const op of step.ops[block]) {
      const left = roots[op.lane], right = roots[op.partner];
      const node = {value:op.result, lane:op.lane, depth, left, right, active:step === f};
      edges.push([left, node], [right, node]);
      nodes.push(node);
      roots[op.lane] = node;
      roots[op.partner] = null;
    }
  }
  let nextX = 0;
  function place(node) {
    if (node.left) {
      place(node.left);
      place(node.right);
      node.x = (node.left.x + node.right.x) / 2;
    } else {
      node.x = 33 + nextX++ * 64;
    }
    node.y = 20 + node.depth * 46;
  }
  roots.filter(Boolean).forEach(place);
  return `<div class="region"><div class="region-title">归约树 / BLOCK ${block}</div>
    <div class="tree"><svg width="${leaves.length * 64 + 4}" height="${42 + depth * 46}" role="img" aria-label="Block ${block} 的归约树">
      ${edges.map(([a, b]) => `<path class="${b.active ? 'active' : ''}" d="M${a.x} ${a.y + 12} C${a.x} ${a.y + 22},${b.x} ${b.y - 22},${b.x} ${b.y - 12}"/>`).join('')}
      ${nodes.map(node => `<g class="${node.active ? 'active' : ''}"><title>Thread ${node.lane} · ${node.value}</title>
        <rect x="${node.x - 30}" y="${node.y - 12}" width="60" height="24" rx="5"/>
        <text x="${node.x}" y="${node.y + 4}" text-anchor="middle">${node.value}</text></g>`).join('')}
    </svg></div></div>`;
}

function raceCanvas(f) {
  const active = f.lane;
  const writes = f.phase === 'write';
  return `<div class="region"><div class="region-title">线程 / 各自要加的值</div>
    <div class="cells">${experiment.values.map((value, lane) =>
      `<span class="slot ${lane === active ? 'active' : ''}"><small>T${lane}</small>+${value}</span>`).join('')}</div></div>
    <div class="compare"><div class="block"><div class="region-title">非原子 · LOAD / ADD / STORE</div>
      <p>全局地址 total</p><div class="slot ${f.lost ? 'risk' : ''} ${active != null ? 'active' : ''}">${f.total}</div>
      <p>${f.lost == null ? (active == null ? '等待执行' : `T${active} · ${{read:'读取', add:'本地计算', write:'写回'}[f.phase]}`)
        : f.lost ? `Lost Update · 少了 ${f.lost}` : '与参考值一致'}</p></div>
    <div class="block"><div class="region-title">原子 · ATOMIC ADD</div>
      <p>全局地址 total</p><div class="slot ${writes ? 'active' : ''}">${f.atomic}</div>
      <p>${f.expected != null ? `参考和 ${f.expected}` : writes ? `T${active} · 完成一次原子更新` : '等待下一次原子更新'}</p></div></div>`;
}

function raceInspector(f) {
  return `<h3>线程寄存器</h3><p>非原子路径的快照和待写回值。</p>${experiment.values.map((value, lane) =>
    `<div class="inspector-row">T${lane} · 加 ${value} · 读取 <code>${f.cache[lane] ?? '—'}</code> · 待写回 <code>${f.pending[lane] ?? '—'}</code></div>`).join('')}`;
}

function drawLinks(f) {
  const area = $('canvas'), svg = area.querySelector('.links');
  if (!svg || !f.links?.length) return;
  const origin = area.getBoundingClientRect();
  svg.setAttribute('width', area.clientWidth);
  svg.setAttribute('height', area.clientHeight);
  svg.style.transform = `translateY(${area.scrollTop}px)`;
  const point = (node, edge) => {
    const box = node.getBoundingClientRect();
    return [box.left + box.width / 2 - origin.left, box[edge] - origin.top];
  };
  svg.innerHTML = f.links.map(([tid, index]) => {
    const cell = $('data-' + index);
    if (!cell) return '';
    const [x1, y1] = point(cell, 'bottom');
    const visible = cell.parentElement.getBoundingClientRect();
    if (y1 < visible.top - origin.top || y1 > visible.bottom - origin.top) return '';
    const [x2, y2] = point($('lane-' + tid), 'top');
    const middle = (y1 + y2) / 2;
    const color = palettes[Math.floor(tid / experiment.threads)];
    return `<path d="M${x1},${y1} C${x1},${middle} ${x2},${middle} ${x2},${y2}" stroke="${color}"/><circle cx="${x1}" cy="${y1}" r="2" fill="${color}"/>`;
  }).join('');
}

function inspector(run, f, selected) {
  const {block, lane} = selected;
  const second = f.stage === 2;
  const width = second ? f.shared[0].length : run.threads;
  const lanes = lane == null ? Array.from({length:width}, (_, i) => i) : [lane];
  const rows = lanes.map(local => {
    const tid = block * width + local;
    const indices = second ? (local < f.partials.length ? [local] : []) : run.indices[tid];
    const values = second ? f.partials : run.values;
    const op = f.ops?.[block]?.find(item => item.lane === local);
    const calculation = op ? `<br>本轮 ${op.left} + ${op.right} = ${op.result}` : '';
    const access = f.links?.find(([global]) => global === tid);
    const reading = access ? `<br>本步读入 <code>[${access[1]}] = ${values[access[1]]}</code>` : '';
    const sharedRead = f.neighbors?.[block]?.[local];
    return `<div class="inspector-row"><b>T${local} · 全局 #${tid}</b><br>
      负责下标 <code>${indices.length ? indices.join(', ') : '—'}</code>
      ${(mode === 'reduce' || lane != null) ? `<br>输入值 <code>${indices.map(i => values[i]).join(', ') || '—'}</code>` : ''}
      ${mode === 'reduce' && lane != null ? `<br>线程小计 <code>${indices.reduce((sum, i) => sum + values[i], 0)}</code>` : ''}
      ${reading}${calculation}${sharedRead == null ? '' : `<br>读右邻居后求和 <code>${sharedRead}</code>`}</div>`;
  }).join('');
  return `<h3>Block ${block}${lane == null ? '' : ` / Thread ${lane}`}</h3>${rows}`;
}

$('nav').addEventListener('click', event => {
  const button = event.target.closest('[data-mode]');
  if (button && mode !== button.dataset.mode) {
    mode = button.dataset.mode;
    setup();
  }
});
$('canvas').addEventListener('click', event => {
  const button = event.target.closest('[data-block]');
  if (!button || !experiment) return;
  selection = {block: Number(button.dataset.block), stage: experiment.frames[cursor].stage || 1};
  if (button.dataset.lane != null) selection.lane = Number(button.dataset.lane);
  lastScroll = $('canvas').scrollTop;
  render();
  const query = selection.lane == null ? `.block-title[data-block="${selection.block}"]` :
    `[data-block="${selection.block}"][data-lane="${selection.lane}"]`;
  $('canvas').querySelector(query)?.focus({preventScroll:true});
});
$('canvas').addEventListener('scroll', () => {
  if (experiment) drawLinks(experiment.frames[cursor]);
}, true);
window.addEventListener('resize', () => {
  if (experiment) drawLinks(experiment.frames[cursor]);
});
$('level').onclick = () => {
  advanced = !advanced;
  $('level').textContent = advanced ? '进阶模式' : '初学者模式';
  $('level').setAttribute('aria-pressed', advanced);
  render();
};
$('settings').onsubmit = event => { event.preventDefault(); load(); };
$('reset').onclick = setup;
$('play').onclick = togglePlay;
$('back').onclick = () => { stop(); seek(cursor - 1); };
$('next').onclick = () => { stop(); seek(cursor + 1); };
$('timeline').oninput = event => { stop(); seek(Number(event.target.value)); };
$('speed').onchange = () => { if (timer) { stop(); togglePlay(); } };
$('gpu').onclick = async () => {
  const output = $('map-result');
  if (!$('map-dialog').open) $('map-dialog').showModal();
  $('gpu').disabled = true;
  output.textContent = '正在调用 NVIDIA 驱动并校验结果…';
  try {
    const data = await post('/api/map', {});
    output.replaceChildren();
    const status = document.createElement('strong');
    status.textContent = data.source === 'gpu' ? '真实 GPU 执行 · 已通过 CPU 校验' : 'CPU 参考结果 · GPU 调用未成功';
    const numbers = document.createElement('p');
    numbers.textContent = `[${data.input.join(', ')}] → [${data.output.join(', ')}]`;
    const log = document.createElement('p');
    log.textContent = data.log;
    output.append(status, numbers, log);
  } catch (error) {
    output.textContent = error.message;
  } finally {
    $('gpu').disabled = false;
  }
};
setup();




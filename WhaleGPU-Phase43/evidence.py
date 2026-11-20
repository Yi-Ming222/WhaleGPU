"""Postprocess existing results without rerunning or removing observations."""
import csv,json,re,sys
from pathlib import Path
from collections import Counter
from xml.sax.saxutils import escape
run=Path(sys.argv[1])
text=(run/'sass.txt').read_text(errors='replace')
funcs=re.split(r'Function\s*:\s*',text)[1:]
with (run/'instructions.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['symbol','opcode','static_count'])
    for s in funcs:
        for op,n in Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\w+\s+)?([A-Z][A-Z0-9_.]+)',s)).items():w.writerow([s.split()[0],op,n])
log=(run/'build.log').read_text(errors='replace')
with (run/'spills.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['symbol','stack_bytes','spill_store_bytes','spill_load_bytes'])
    w.writerows(re.findall(r'Function properties for (\S+)\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log))
rows=list(csv.DictReader((run/'summary.csv').open()))
cases=list(dict.fromkeys(r['case'] for r in rows))
for m in ['kernel_ms','e2e_ms']:
    panel=[]
    for tag in ['first','repeat']:
        zs={v:[float(next(r['median'] for r in rows if (r['run'],r['metric'],r['variant'],r['case'])==(tag,m,v,c))) for c in cases] for v in ['split','fused']}
        top=max(map(max,zs.values()));panel.append(f'<text x="5" y="20">{tag} · median {m} · axis 0 to {top:.4g} ms</text>')
        for v,color in [('split','#124db3'),('fused','#b63422')]:
            pts=' '.join(f'{50+630*i/max(1,len(cases)-1):.2f},{180-145*y/top:.2f}' for i,y in enumerate(zs[v]))
            panel.append(f'<polyline points="{pts}" fill="none" stroke="{color}"/><text x="{400 if v=="split" else 500}" y="38" fill="{color}">{v}</text>')
        panel.append('<path d="M50 32 V180 H680" fill="none" stroke="black"/>')
        for i in range(len(cases)):panel.append(f'<text x="{50+630*i/max(1,len(cases)-1):.1f}" y="198">{i+1}</text>')
        if tag=='first':panel.append('</g><g transform="translate(0 210)">')
    (run/(m+'.svg')).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="740" height="430"><g>'+''.join(panel)+'</g></svg>')
with (run/'REPORT.md').open('a',encoding='utf8') as f:
    f.write('\n\n## Correctness and evidence\n\n')
    for tag in ('validate','first','repeat'):
        checks=list(csv.DictReader((run/tag/'checks.csv').open()));f.write(f'- {tag} completed {len(checks)} full-output checks. Maximum absolute error across cases {max(float(r["max_abs"]) for r in checks):.7g}.\n')
    f.write('\nThe kernel interval includes every launch in the chain. E2E includes upload, execution and synchronous download. Both exclude allocation. Cache condition is recorded by run name (ring = rotating device addresses, otherwise reused). Achieved occupancy, measured DRAM traffic and synchronization diagnostics are unavailable unless separately collected. Theoretical residency and launch geometry are in first/resources.csv. Static code evidence is in instructions.csv and spills.csv.\n')
    for m in ['kernel_ms','e2e_ms']:f.write(f'\n![{m}]({m}.svg)\n')
    for i,c in enumerate(cases):f.write(f'\n{i+1}. {c}')

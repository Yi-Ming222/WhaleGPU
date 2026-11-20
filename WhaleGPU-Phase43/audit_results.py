"""Verify raw sample coverage and hashes, then join resources to exact ptxas entries."""
import csv,json,math,random,statistics as st,sys
from pathlib import Path
from history import sha
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
prefix={'register_stats':'register_rowILb0','register_fused':'register_rowILb1',
        'local_tile_multi':'local_tileILb0','local_tile_single':'local_tileILb1'}
def symbol_match(kernel,case,s):
    if prefix.get(kernel,kernel) not in s:return False
    if kernel in ('block_gemm','pipeline_affine'):return ('bfloat16' in s)==case.startswith('BF16') and 'Lb1E' in s
    if kernel in ('register_gemm','register_affine'):return 'ILi64ELi4ELb0' in s
    if kernel=='two_pass':return 'Lb0E' in s
    return True
for arg in sys.argv[1:]:
    run=Path(arg)
    for manifest,folder in [('source_sha256.json',run/'snapshot'),('binary.json',run)]:
        assert all(sha(folder/k)==v for k,v in json.loads((run/manifest).read_text()).items())
    checks=read(run/'validate/checks.csv')
    assert checks and all(math.isfinite(float(r['max_abs'])) and (int(r['n'])==0 or math.isfinite(float(r['rmse']))) for r in checks)
    spills=read(run/'spills.csv');merged=[];pairs={};rng=random.Random(43)
    for tag in ['first','repeat']:
        raw=read(run/tag/'samples.csv');resources=read(run/tag/'resources.csv')
        device=read(run/tag/'device.csv')[0]
        for case in dict.fromkeys(r['case'] for r in raw):
            subset=[r for r in raw if r['case']==case]
            variants=set(r['variant'] for r in subset)
            assert len(subset)==21*len(variants)
            for i in range(21):
                rr=[r for r in subset if int(r['pair'])==i]
                assert {r['variant'] for r in rr}==variants and sorted(int(r['order']) for r in rr)==list(range(len(rr)))
            for row in [r for r in resources if r['case']==case and r['variant'] in variants]:
                matches=[s for s in spills if symbol_match(row['kernel'],case,s['symbol'])]
                assert len(matches)==1,(row,matches)
                warps=math.ceil(int(row['threads'])/int(device['warp']))
                occ=warps*int(row['blocks_per_sm'])/(int(device['max_threads_per_sm'])/int(device['warp']))
                assert abs(float(row['theoretical_occupancy'])-occ)<1e-10
                merged.append(dict(process=tag,**row,**matches[0],achieved_occupancy='NOT_COLLECTED'))
            for metric in ('kernel_ms','e2e_ms'):
                def values(id):return [float(r[metric]) for r in sorted((r for r in subset if r['variant']==id),key=lambda r:int(r['pair']))]
                x,y=values('split'),values('fused')
                assert all(math.isfinite(t) and t>0 for t in x+y)
                z=[a/b for a,b in zip(x,y)]
                boot=sorted(st.median(rng.choices(z,k=21)) for _ in range(2000))
                pairs.setdefault((case,metric),[]).append((st.median(z),boot[49]*.025+boot[50]*.975,boot[1949]*.975+boot[1950]*.025))
    write(run/'resources_verified.csv',merged)
    rows=[]
    for (case,metric),((x,a,b),(y,c,d)) in sorted(pairs.items()):
        verdict='gain' if min(a,c)>1 else 'regression' if max(b,d)<1 else 'uncertain'
        rows.append(dict(case=case,metric=metric,first=x,lo1=a,hi1=b,repeat=y,lo2=c,hi2=d,verdict=verdict))
    write(run/'paired_verdicts.csv',rows)

"""Cross-check raw files, selection, source integrity, and comparison conditions."""
from pathlib import Path


def audit(run):
    from run import rows, read, save, sha, key, ROOT, phase1_hashes
    counts={}
    hardware=read(run/'engine'/'hardware.json')
    eligible={key(x) for x in read(run/'validate'/'eligible.json')}
    for stage in ['engine','validate','matrix','tune','confirm']:
        directory=run/stage
        plan=rows(directory/'plan.csv')
        ids={x['id']:x for x in plan}
        if len(ids)!=len(plan): raise ValueError('Duplicate job ids')
        checks=rows(directory/'correctness.csv')
        if {x['id'] for x in checks}!=set(ids) or len(checks)!=len(plan):
            raise ValueError('Correctness coverage mismatch')
        if any(x['correct']!='PASS' for x in checks): raise ValueError('Failed correctness')
        hw=read(directory/'hardware.json')
        for field in ['gpu','cc','sm_count','l2_bytes','nvcc','runtime','msvc']:
            if hw[field]!=hardware[field]: raise ValueError('Mixed hardware/compiler')
        samples=rows(directory/'samples.csv')
        if stage!='validate':
            expected=sum(int(x['rounds']) for x in plan)
            if len(samples)!=expected: raise ValueError('Sample count mismatch')
            if any(x['correct']!='PASS' for x in samples): raise ValueError('Bad measured result')
        if stage=='tune' and any(key(x) not in eligible for x in plan):
            raise ValueError('Unvalidated candidate entered search')
        counts[stage]=dict(jobs=len(plan),samples=len(samples))
    candidates=rows(run/'tune'/'summary.csv')
    winners=read(run/'winners.json')
    for winner in winners:
        matching=[x for x in candidates if x['op']==winner['op'] and x['n']==winner['n']]
        if winner['id']!=min(matching,key=lambda x:float(x['kernel_median']))['id']:
            raise ValueError('Winner selection mismatch')
    confirm=rows(run/'confirm'/'plan.csv')
    groups={x['group'] for x in confirm}
    for group in groups:
        triplet=[x for x in confirm if x['group']==group]
        if len(triplet)!=3: raise ValueError('Incomplete confirmation pair')
        for field in ['op','n','allocation','pattern','rounds','warmups','repeats']:
            if len({x[field] for x in triplet})!=1: raise ValueError('Unfair comparison conditions')
        selected=[x for x in triplet if not x['algorithm'].startswith('phase1_')]
        if len(selected)!=1 or not any(key(selected[0])==key(w) and selected[0]['n']==w['n'] for w in winners):
            raise ValueError('Confirmation does not match winner')
    for file in (ROOT/'src').rglob('*'):
        if file.is_file() and sha(file)!=sha(run/'source'/'phase2'/'src'/file.relative_to(ROOT/'src')):
            raise ValueError('Kernel or benchmark source differs from tested snapshot')
    if phase1_hashes()!=read(run/'phase1-before.json'):
        raise ValueError('Phase 1 source modified')
    if sha(run/'build'/'whalegpu_bench.exe')!=read(run/'binary-sha256.json')['whalegpu_bench.exe']:
        raise ValueError('Binary changed')
    regression=(run/'phase1-regression.log').read_text(encoding='utf-8',errors='replace')
    if 'PASS stage=all cases=672 failed=0' not in regression:
        raise ValueError('Phase 1 full regression not verified')
    result=dict(status='PASS',counts=counts,independent_pair_groups=len(groups),source_and_binary_unchanged=True,
                checks=['raw counts','correctness coverage','candidate eligibility','winner selection','paired conditions',
                        'same hardware/compiler','source and binary hashes','Phase 1 672-case regression'])
    save(run/'experiment-audit.json',result)
    return result

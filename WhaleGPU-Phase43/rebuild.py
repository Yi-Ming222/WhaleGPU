"""Independently rebuild frozen snapshots and compare full correctness CSVs."""
import json,os,shutil,subprocess as sp,sys,uuid
from pathlib import Path
from history import ROOT,manifest,sha
hist=manifest()
out=ROOT/'rebuild'/uuid.uuid4().hex[:8];out.mkdir(parents=True)
for arg in sys.argv[1:]:
    run=Path(arg).resolve();exp=run.name[0];work=out/run.name
    shutil.copytree(run/'snapshot',work/'snapshot');shutil.copyfile(run/'build.cmd',work/'build.cmd')
    for k,v in json.loads((run/'source_sha256.json').read_text()).items():assert sha(work/'snapshot'/k)==v
    env=os.environ.copy();env.pop('WHALEGPU43_ROTATE',None);env['OMP_NUM_THREADS']='8'
    with (work/'build.log').open('wb') as f:sp.run(['cmd','/d','/c','..\\..\\build.cmd'],cwd=work/'snapshot'/ROOT.name,env=env,stdout=f,stderr=sp.STDOUT,check=True)
    (work/'validate').mkdir();exe=work/(exp+'.exe')
    with (work/'validate.log').open('wb') as f:sp.run([exe,'validate','validate','43'],cwd=work,env=env,stdout=f,stderr=sp.STDOUT,check=True)
    original=sha(run/'validate/checks.csv');actual=sha(work/'validate/checks.csv')
    assert original==actual
    (work/'verified.json').write_text(json.dumps(dict(source=run.name,binary_sha256=sha(exe),original_binary_sha256=sha(run/exe.name),correctness_sha256=actual,correctness_identical=True),indent=2))
assert hist==manifest()
print(out)

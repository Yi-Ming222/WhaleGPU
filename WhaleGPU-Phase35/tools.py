from pathlib import Path
import urllib.request, json, hashlib, zipfile, io
root=Path(__file__).resolve().parent/'tools'
root.mkdir(exist_ok=True)
base='https://developer.download.nvidia.com/compute/cuda/redist/'
manifest=json.load(urllib.request.urlopen(base+'redistrib_13.0.0.json',timeout=40))
for name in ['cuda_cuobjdump','cuda_nvdisasm']:
    pkg=manifest[name]['windows-x86_64']
    data=urllib.request.urlopen(base+pkg['relative_path'],timeout=90).read()
    assert hashlib.sha256(data).hexdigest()==pkg['sha256']
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.namelist():
            if member.endswith(('.exe','.dll','LICENSE')): (root/Path(member).name).write_bytes(archive.read(member))
    (root/(name+'.json')).write_text(json.dumps(pkg,indent=2))

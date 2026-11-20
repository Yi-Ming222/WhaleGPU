from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
OLD=sorted(p for p in ROOT.parent.glob("WhaleGPU-Phase*") if p!=ROOT)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def manifest():
    entries={}
    audit=(ROOT/'PHASE43_AUDIT.md').read_text(encoding='utf8')
    for folder in OLD:
        digest=hashlib.sha256()
        for file in sorted((p for p in folder.rglob('*') if p.is_file()),key=lambda p:p.relative_to(folder).as_posix()):
            rel=file.relative_to(folder).as_posix()
            value=sha(file);entries[folder.name+'/'+rel]=value
            digest.update((rel+'\0'+value+'\n').encode())
        if digest.hexdigest() not in audit:raise RuntimeError('Historical tree mismatch '+folder.name)
    return entries
if __name__=='__main__':
    print(len(manifest()),'files match approved tree anchors')

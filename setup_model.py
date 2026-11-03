"""Download the pinned full-precision ONNX export; check SHA-256 before activation."""
from pathlib import Path
from urllib.request import urlopen
import hashlib,shutil
f=Path(__file__).resolve().parent/'models/dinov2-small.onnx'
digest='83141175ec78b4ff9a2bb58a4c7c264ba0054d1c2e122e5a8114b79a8d4179ea'
url='https://huggingface.co/Xenova/dinov2-small/resolve/c2bb04a51fab207c420665f1946016107bffc701/onnx/model.onnx'
def valid(p):return p.is_file() and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==digest
if not valid(f):
    f.parent.mkdir(exist_ok=True);tmp=f.with_suffix('.part')
    with urlopen(url,timeout=120) as source,tmp.open('wb') as out:shutil.copyfileobj(source,out)
    if not valid(tmp):raise ValueError('Encoder checksum mismatch')
    tmp.replace(f)
print('DINOv2 encoder ready.')


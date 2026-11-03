from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
CLASSES = ['bottle', 'cup', 'pen']
def prepare(im):
    im = ImageOps.exif_transpose(im).convert('RGB')
    im.thumbnail((224, 224), Image.Resampling.LANCZOS)
    frame = Image.new('RGB',(224,224),(124,116,104))
    frame.paste(im, ((224-im.width)//2,(224-im.height)//2))
    return frame

class Encoder:
    def __init__(self):
        options=ort.SessionOptions()
        options.intra_op_num_threads=4
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(ROOT/'models/dinov2-small.onnx'), options, providers=['CPUExecutionProvider'])
    def __call__(self,frame):
        x=np.asarray(frame, dtype=np.float32)/255.
        x=(x-np.float32([.485,.456,.406]))/np.float32([.229,.224,.225])
        h=self.session.run(None, {'pixel_values':x.transpose(2,0,1)[None]})[0][0,0]
        return h/np.linalg.norm(h)

def softmax(x):
    z=np.exp(x-x.max(axis=-1,keepdims=True))
    return z/z.sum(axis=-1,keepdims=True)


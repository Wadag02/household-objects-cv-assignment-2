"""Local-only web demo. Run python app.py and visit http://127.0.0.1:7860."""
import io, json, platform, statistics, time, threading, argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from mimetypes import guess_type
from vision import *

class Classifier:
    def __init__(self):
        self.encode=Encoder();self.params=np.load(ROOT/'models/head.npz')
        self.gallery=json.loads((ROOT/'models/gallery.json').read_text('utf8'))
        self.latency=deque(maxlen=30); self.lock=threading.Lock()
    def predict(self,raw):
        with self.lock:
            start=time.perf_counter()
            with Image.open(io.BytesIO(raw)) as im:
                if im.width*im.height>50_000_000:raise ValueError('Image exceeds 50 megapixels.')
                z=self.encode(prepare(im))
            h=self.params
            probs=softmax(z@h['w']+h['b']);sim=h['index']@z
            neighbors=[dict(**self.gallery[j],similarity=float(sim[j])) for j in np.argsort(-sim)[:4]]
            ms=(time.perf_counter()-start)*1000;self.latency.append(ms)
            return dict(label=CLASSES[probs.argmax()],scores=dict(zip(CLASSES,probs)),neighbors=neighbors,ms=ms,mean=statistics.mean(self.latency),count=len(self.latency),device=platform.system()+' / '+platform.machine()+' / ONNX Runtime CPU')
class Handler(BaseHTTPRequestHandler):
    def reply(self,code,kind,body):
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/':path='/index.html'
        base=(ROOT/'web').resolve();f=(base/path.lstrip('/')).resolve()
        if base not in f.parents or not f.is_file():return self.send_error(404)
        self.reply(200,guess_type(f.name)[0] or 'application/octet-stream',f.read_bytes())
    def do_POST(self):
        if self.path!='/predict':return self.send_error(404)
        try:
            n=int(self.headers.get('Content-Length',0))
            if not 0<n<=25_000_000:raise ValueError('Choose an image smaller than 25 MB.')
            res=self.server.model.predict(self.rfile.read(n))
            self.reply(200,'application/json',json.dumps(res).encode())
        except (OSError,ValueError,Image.DecompressionBombError):self.reply(400,'application/json',b'{"error":"Use a readable JPEG, PNG or WebP under 25 MB and 50 MP."}')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=7860);args=ap.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.model=Classifier()
    print(f'Open http://127.0.0.1:{args.port}',flush=True);server.serve_forever()


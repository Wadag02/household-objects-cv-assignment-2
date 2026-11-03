"""Reproduce the experiment: python train.py /path/to/Dataset"""
import csv, json, hashlib, sys, time, platform, unicodedata
from collections import Counter
from PIL import Image, ImageOps
from vision import *
from importlib.metadata import version

def read(file):
    with open(ROOT/file,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
    with open(ROOT/'results'/name,'w',encoding='utf8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
def scores(y,p):
    cm=np.zeros((3,3),int);np.add.at(cm,(y,p),1)
    f=2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)
    return float(np.trace(cm)/len(y)),float(f.mean()),cm.tolist()
def main(data):
    rows=read('evidence/dataset_manifest.csv'); a1=read('evidence/test_predictions.csv')
    assert len(rows)==len({r['sha256'] for r in rows})==420
    assert sorted(Counter(r['object_id'] for r in rows).values())==[20]*21
    assert len(set((r['object_id'],r['split']) for r in rows))==21
    assert Counter(r['split'] for r in rows)=={'train':300,'val':60,'test':60}
    assert {r['sha256'] for r in rows if r['split']=='test'}=={r['sha256'] for r in a1}
    encoder=Encoder()
    xs=[];aug=[];gallery=[];times=[]
    for name in ['gallery','examples']: (ROOT/'web'/name).mkdir(parents=True,exist_ok=True)
    norm=lambda s:unicodedata.normalize('NFC',s)
    paths={norm(p.relative_to(data).as_posix()):p for p in data.rglob('*') if p.is_file()}
    for i,r in enumerate(rows):
        path=paths[norm(r['path'].split('/Dataset/')[1])]
        with Image.open(path) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB')
            digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if digest!=r['sha256']: raise ValueError('A1 pixel hash mismatch: '+str(path))
            frame=prepare(rgb)
        start=time.perf_counter();v=encoder(frame);times.append((time.perf_counter()-start)*1000);xs.append(v)
        if r['split']=='train':
            aug.append(encoder(ImageOps.mirror(frame)))
            name=f'gallery/{i:03}.jpg';frame.save(ROOT/'web'/name,quality=87)
            gallery.append({'image':name,'class':r['class'],'object_id':r['object_id'],'file':path.name})
        else:frame.save(ROOT/'web/examples'/(path.stem+'.png'))
        if i%60==0:print('Verified and encoded:',i+1,flush=True)
    X=np.array(xs); y=np.array([int(r['label']) for r in rows]); split=np.array([r['split'] for r in rows])
    tr,va,te=(split==s for s in ('train','val','test'))
    history=[];heads={}
    for mode in ['original','mirror']:
        xx=X[tr] if mode=='original' else np.concatenate((X[tr],np.array(aug)))
        yy=y[tr] if mode=='original' else np.tile(y[tr],2)
        w=np.zeros((X.shape[1],3));b=np.zeros(3); best=-1
        for epoch in range(1,181):
            prob=softmax(xx@w+b);g=(prob-np.eye(3)[yy])/len(yy)
            w-=5*(xx.T@g+.0005*w);b-=5*g.sum(0)
            sp=softmax(xx@w+b);vp=softmax(X[va]@w+b)
            loss=lambda a,t:float(-np.log(a[np.arange(len(t)),t]).mean())
            acc,f1,_=scores(y[va],vp.argmax(1))
            history.append(dict(mode=mode,epoch=epoch,train_loss=loss(sp,yy),val_loss=loss(vp,y[va]),val_accuracy=acc,val_f1=f1))
            if f1>best:best=f1;heads[mode]=(best,epoch,w.copy(),b.copy())
    selected=max(heads,key=lambda m:heads[m][0])
    save('training_curves.csv',history)
    preds={}; metrics={}; lookup={r['sha256']:r for r in a1}
    for name,col in [('A1 pixels','baseline_prediction'),('A1 ResNet18','resnet18_prediction')]:
        preds[name]=np.array([CLASSES.index(lookup[r['sha256']][col]) for r in rows if r['split']=='test'])
    for m,(_,epoch,w,b) in heads.items():preds[m]=(X[te]@w+b).argmax(1)
    for m,p in preds.items():
        acc,f,cm=scores(y[te],p);metrics[m]=dict(accuracy=acc,macro_f1=f,confusion=cm,correct=int((p==y[te]).sum()))
        if m in heads:metrics[m].update(val_f1=heads[m][0],epoch=heads[m][1])
    _,_,w,b=heads[selected]
    np.savez(ROOT/'models/head.npz',w=w,b=b,index=X[tr])
    (ROOT/'models/gallery.json').write_text(json.dumps(gallery,ensure_ascii=False),encoding='utf8')
    testrows=[]
    for i,r in enumerate(r for r in rows if r['split']=='test'):
        v=X[te][i]; ss=X[tr]@v; nearest=np.argsort(-ss)[:4]
        testrows.append(dict(file=Path(r['path']).name,truth=r['class'],object_id=r['object_id'],sha256=r['sha256'],**{m:CLASSES[p[i]] for m,p in preds.items()},neighbors=';'.join(gallery[j]['file'] for j in nearest)))
    save('test_predictions.csv',testrows)
    out=dict(selected=selected,metrics=metrics,mean_encoder_ms=float(np.mean(times[1:])),versions={p:version(p) for p in ['onnxruntime','numpy','Pillow']},python=platform.python_version(),platform=platform.platform())
    (ROOT/'results/summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__': main(Path(sys.argv[1]))


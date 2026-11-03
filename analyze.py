from pathlib import Path
import sys,csv,json,io,statistics,platform
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from app import Classifier,CLASSES
from PIL import Image
def rows(path):return list(csv.DictReader(path.open(encoding='utf8')))
rd=P/'results'; summary=json.loads((rd/'summary.json').read_text());sel=summary['selected'];tests=rows(rd/'test_predictions.csv')
# Prefer a paired correction and a natural error on the deployed model.
good=next((r for r in tests if r[sel]==r['truth'] and r['A1 ResNet18']!=r['truth']),next(r for r in tests if r[sel]==r['truth']))
bad=next((r for r in tests if r[sel]!=r['truth']),None)
def example(r):return 'examples/'+Path(r['file']).stem+'.png'
cases=[dict(file=good['file'],truth=good['truth'],image=example(good),kind='Held-out original',a1=good['A1 ResNet18'])]
model=Classifier()
if bad:cases.append(dict(file=bad['file'],truth=bad['truth'],image=example(bad),kind='Held-out original',a1=bad['A1 ResNet18']))
else:
 for r in tests:
  if len(cases)>1:break
  original=Image.open(P/'web'/example(r)).convert('RGB')
  for percent in [.5,.7,.85,.95]:
   img=original.copy(); margin=int((1-percent)*112);img.paste((124,116,104),(margin,margin,224-margin,224-margin))
   buf=io.BytesIO();img.save(buf,format='PNG')
   if model.predict(buf.getvalue())['label']!=r['truth']:
    img.save(P/'web/examples/stress.png');cases.append(dict(file=r['file']+' (occluded)',truth=r['truth'],image='examples/stress.png',kind=f'Edited stress case: central {100*percent:.0f}% of width and height masked; excluded from metrics',a1=None));break
assert len(cases)==2
(P/'web/cases.json').write_text(json.dumps(cases,indent=2))
trial=[]
for r in tests:
 raw=(P/'web'/example(r)).read_bytes();d=model.predict(raw);assert d['label']==r[sel]; trial.append(d['ms'])
bench=dict(mean=statistics.mean(trial),median=statistics.median(trial),n=len(trial),input='224x224 PNG model inputs; server decode + preprocess + encoder + head + retrieval, warm model; no network or browser',cpu=platform.processor() or platform.machine())
(rd/'latency.json').write_text(json.dumps(bench,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#274943','axes.titleweight':'bold','figure.facecolor':'white'})
colors=['#0c6a5b','#aa562e']
hist=rows(rd/'training_curves.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.1))
for j,m in enumerate(['original','mirror']):
 h=[r for r in hist if r['mode']==m];x=[int(r['epoch']) for r in h]
 axs[0].plot(x,[float(r['train_loss']) for r in h],color=colors[j],label=m)
 axs[1].plot(x,[float(r['val_f1']) for r in h],color=colors[j],label=m)
 e=summary['metrics'][m]['epoch'];axs[1].scatter([e],[summary['metrics'][m]['val_f1']],color=colors[j])
for a,title in zip(axs,['Training cross entropy','Validation macro-F1']):
 a.set(title=title,xlabel='Full-batch update');a.legend(frameon=False);a.grid(axis='y',alpha=.18)
fig.tight_layout();fig.savefig(rd/'curves.png',dpi=170);plt.close()
fig,axs=plt.subplots(1,3,figsize=(9,3.1))
for ax,m in zip(axs,['A1 pixels','A1 ResNet18',sel]):
 cm=np.array(summary['metrics'][m]['confusion']);ax.imshow(cm,cmap='BuGn',vmin=0,vmax=20)
 for (i,j),v in np.ndenumerate(cm):ax.text(j,i,v,va='center',ha='center',color='white' if v>12 else '#183b34',fontsize=13)
 ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,title=m if m.startswith('A1') else 'DINOv2 selected',xlabel='Predicted');ax.tick_params(length=0)
axs[0].set_ylabel('True class');fig.tight_layout();fig.savefig(rd/'confusions.png',dpi=170);plt.close()
details=[]
for i,c in enumerate(cases):
 d=model.predict((P/'web'/c['image']).read_bytes());details.append(d)
 fig,axs=plt.subplots(1,5,figsize=(9,2.45))
 for ax,im,title in zip(axs,[c['image']]+[n['image'] for n in d['neighbors']],['Query: '+c['truth']]+[f"{n['class']}  {n['similarity']:.3f}" for n in d['neighbors']]):
  ax.imshow(Image.open(P/'web'/im));ax.set_title(title,fontsize=11);ax.axis('off')
 fig.tight_layout();fig.savefig(rd/f'case{i+1}.jpg',dpi=160,bbox_inches='tight');plt.close()
(rd/'case_details.json').write_text(json.dumps(details,indent=2))
print(json.dumps([summary,bench,cases],ensure_ascii=True))


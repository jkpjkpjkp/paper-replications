#!/usr/bin/env python3
"""Audit the exact published processed graph; no model fitting."""
import hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
os.environ['DGLBACKEND']='pytorch';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch,dgl,numpy as np
from dgl.data.utils import load_graphs
from importlib.metadata import version
path=R/'data/tfinance';m=json.loads((R/'sources/data-manifest.json').read_text())
with path.open('rb') as h:assert hashlib.file_digest(h,'sha256').hexdigest()==m['sha256']
print('Loading exact graph',flush=True);g=load_graphs(str(path))[0][0]
labels=g.ndata['label'];x=g.ndata['feature'];n=g.num_nodes()
print('Loaded',n,g.num_edges(),list(g.ndata),flush=True)
assert x.shape==(n,10) and labels.shape==(n,) and torch.isfinite(x).all()
assert set(labels.unique().tolist())=={0,1}
splits=[]
for t in range(g.ndata['train_masks'].shape[1]):
 masks=[g.ndata[k][:,t].bool() for k in ['train_masks','val_masks','test_masks']]
 a,b,c=masks;overlap=int((a&b).sum()+(a&c).sum()+(b&c).sum());uncovered=int((~(a|b|c)).sum())
 splits.append({'column':t,'role':'fully_supervised' if t<10 else 'semi_supervised','counts':dict(zip(['train','validation','test'],[int(z.sum()) for z in masks])),'fraud_counts':dict(zip(['train','validation','test'],[int(labels[z].sum()) for z in masks])),'fractions':[float(z.float().mean()) for z in masks],'overlap_memberships':overlap,'uncovered_nodes':uncovered,'sha256':{k:hashlib.sha256(z.numpy().tobytes()).hexdigest() for k,z in zip(['train','validation','test'],masks)}})
print('Full label homophily in chunks',flush=True)
src,dst=g.edges();same=0;train_same=[0]*10;train_edges=[0]*10
for start in range(0,len(src),1000000):
 s=src[start:start+1000000];d=dst[start:start+1000000];eq=labels[s]==labels[d];same+=int(eq.sum())
 for j in range(10):
  mask=g.ndata['train_masks'][:,j].bool();sel=mask[s]&mask[d];train_edges[j]+=int(sel.sum());train_same[j]+=int(eq[sel].sum())
report={'paper_id':429,'created_at':datetime.now(timezone.utc).isoformat(),'graph_path':str(path),'sha256':m['sha256'],'nodes':n,'edges':g.num_edges(),'ndata':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in g.ndata.items()},'labels':{str(int(y)):int((labels==y).sum()) for y in labels.unique()},'feature_finite':bool(torch.isfinite(x).all()),'feature_means':x.mean(0).tolist(),'feature_stds':x.std(0,unbiased=False).tolist(),'splits':splits,'fully_supervised_masks_disjoint_exhaustive':all(z['overlap_memberships']==z['uncovered_nodes']==0 for z in splits[:10]),'all_label_homophily':same/g.num_edges(),'train_only_edge_homophily':[{ 'trial':j,'observed_train_train_edges':train_edges[j],'homophily':train_same[j]/train_edges[j] if train_edges[j] else None} for j in range(10)],'sample_dates':'not serialized; no chronological claim','dependencies':{'python':sys.version,'torch':torch.__version__,'dgl':dgl.__version__,'numpy':np.__version__,'catboost':version('catboost')},'environment_reuse':'Runtime Python environment; exact imported package versions are recorded above','paper_comparison':{'paper_nodes':39357,'paper_edges':'42.0M','paper_d':10,'paper_split_claim':'40/20/40; actual processed masks explicitly reported here'}}
p=R/'results/data-audit.json';p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
print(json.dumps({k:report[k] for k in ['nodes','edges','labels','fully_supervised_masks_disjoint_exhaustive','all_label_homophily']},indent=2));print('First split',splits[0],flush=True)

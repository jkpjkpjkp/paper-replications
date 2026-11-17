#!/usr/bin/env python3
"""Guard-only checkpoint/prediction/metric and actual-logit decomposition audit."""
import json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'official'));os.environ['DGLBACKEND']='pytorch'
import numpy as np,torch
from dgl.data.utils import load_graphs
from sklearn.metrics import roc_auc_score,average_precision_score
from models.gnn import FraudGNAM
from evaluate_faithfulness import compute_component_vectors,logit_from_components,loco_faithfulness,topk_sufficiency,topk_comprehensiveness
torch.set_num_threads(2);torch.set_num_interop_threads(1)
G=load_graphs(str(R/'data/tfinance'))[0][0]
rows=[]
for trial in range(10):
 folder=R/f'results/local-main/trial{trial}';rec=json.loads((folder/'result.json').read_text());z=np.load(folder/'best-test-predictions.npz');y=z['labels'];p=z['probabilities'];k=int(y.sum())
 assert np.array_equal(z['node_ids'],np.flatnonzero(G.ndata['test_masks'][:,trial].numpy()))
 assert np.array_equal(y,G.ndata['label'][torch.from_numpy(z['node_ids'])].numpy())
 metrics={'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'RecK':float(y[p.argsort()[-k:]].sum()/k)}
 assert all(abs(metrics[m]-rec['metrics'][m])<1e-12 for m in metrics)
 rows.append({'trial':trial,'seed':rec['seed'],'saved_predictions_metrics_verified':True,'metrics':metrics})
# Independently reconstruct predictions from the best epoch checkpoint for trial0.
folder=R/'results/local-main/trial0';checkpoint=torch.load(folder/'best-validation-model.pt',map_location='cpu',weights_only=False)
model=FraudGNAM(**checkpoint['model_config']);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
with torch.no_grad():logits=model(G);pred=logits.softmax(1)[:,1]
z=np.load(folder/'best-test-predictions.npz');idx=torch.from_numpy(z['node_ids'])
maxdiff=float(np.max(np.abs(pred[idx].numpy()-z['probabilities'])))
assert maxdiff<1e-6
components=compute_component_vectors(model,G);reconstructed=logit_from_components(components);actual=logits[:,1]-logits[:,0]
logit_error=float((reconstructed-actual).abs().max());assert logit_error<1e-4
mask=G.ndata['test_masks'][:,0].bool();labels=G.ndata['label']
faith={'native_loco':loco_faithfulness(components,mask),'native_per_node_topk_sufficiency':topk_sufficiency(components,labels,mask),'native_per_node_topk_comprehensiveness':topk_comprehensiveness(components,labels,mask),'actual_forward_logit_max_abs_error':logit_error,'checkpoint_prediction_max_abs_difference':maxdiff,'checkpoint_scope':'best validation epoch; source explanation scripts used final state, so comparison must disclose checkpoint convention','topk_scope':'Source uses per-node topk; paper AppJ prose describes ranking mean absolute contribution across test nodes. Both conventions must remain distinct.'}
report={'paper_id':429,'all10_predictions_audited':rows,'trial0_checkpoint_and_faithfulness':faith,'no_return_test':'Nontrading classifier; market_neutral_nocost not applicable.'}
pth=R/'results/local-main-validation.json';tmp=pth.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(pth);print(json.dumps({'prediction_checks':len(rows),'checkpoint_maxdiff':maxdiff,'decomposition_maxdiff':logit_error,'output':str(pth)},indent=2))

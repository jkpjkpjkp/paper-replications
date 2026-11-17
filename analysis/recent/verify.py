"""Audit recent test boundaries, source hashes, weights, and a dollar cash ledger."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hermes.metrics import PORTFOLIO_PROTOCOL,evaluate


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit_cash_ledger(d):
    holdings=np.zeros(d['portfolio_weights'].shape[1]);cash=1.;nav=1.
    for t,(weights,price,total) in enumerate(zip(d['portfolio_weights'],d['target'],d['pnl_target'])):
        before=nav
        fee=float(d['entry_fee_fraction'][t])*before
        target=weights*(before-fee)
        orders=target-holdings
        np.testing.assert_allclose(fee,.0005*np.maximum(orders,0).sum(),rtol=1e-10,atol=1e-12)
        cash=cash-orders.sum()-fee
        holdings=target*(1+price.astype(float))
        cash+=target @ (total.astype(float)-price.astype(float))
        nav=cash+holdings.sum()
        if t==len(d['dates'])-1:
            cover_fee=.0005*np.maximum(-holdings,0).sum()
            cash+=holdings.sum()-cover_fee;holdings[:]=0;nav=cash
        np.testing.assert_allclose(nav/before-1,d['net_portfolio_return'][t],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(nav,np.prod(1+d['net_portfolio_return']),rtol=1e-10)


def main():
    manifest=json.loads((ROOT/'data/recent/manifest.json').read_text())
    source=ROOT.parent / 'crsp-daily/data/crsp_a_stock/dsf_v2'
    for record in manifest['source_receipts']:
        assert sha(source/f'{record["year"]}.parquet')==record['receipt']['sha256']
    assert manifest['missing_post_anchor_eligible_returns']==0
    root=ROOT/'results/recent';paths=sorted(root.glob('*/metrics.json'))
    assert len(paths)==10
    checks=[]
    for path in paths:
        run=path.parent;result=json.loads(path.read_text());d=np.load(run/'test_predictions.npz')
        assert result['evaluation_protocol']==PORTFOLIO_PROTOCOL
        weights=d['portfolio_weights']
        np.testing.assert_allclose(weights.sum(1),0,atol=1e-12)
        np.testing.assert_allclose(np.maximum(weights,0).sum(1),.5,atol=1e-12)
        np.testing.assert_allclose(np.minimum(weights,0).sum(1),-.5,atol=1e-12)
        assert ((weights!=0).sum(1)==10).all()
        assert not np.any((weights!=0)&~d['valid'])
        metrics,_,returns=evaluate(d['prediction'],d['target'],d['valid'],5,d['pnl_target'])
        for key,value in metrics.items():
            if value is None:assert result['test'][key] is None
            else:np.testing.assert_allclose(value,result['test'][key],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(returns,d['portfolio_return'])
        audit_cash_ledger(d)
        if run.name.startswith('rolling'):
            config=json.loads((run/'config.json').read_text());data=np.load(config['data'])
            np.testing.assert_array_equal(d['dates'],data['dates'][config['val_end']:])
            np.testing.assert_array_equal(d['valid'],data['valid'][config['val_end']:])
            np.testing.assert_array_equal(d['target'],data['returns'][config['val_end']:])
            np.testing.assert_array_equal(d['pnl_target'],data['pnl_returns'][config['val_end']:])
            np.testing.assert_allclose(data['features'][:config['train_end']].mean(0),0,atol=2e-5)
            history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
            assert min(history,key=lambda x:x['val_loss'])['epoch']==result['best_epoch']
            checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            assert checkpoint['epoch']==result['best_epoch']
            assert data['dates'][config['train_end']-1]<data['dates'][config['train_end']]<data['dates'][config['val_end']]
            model_sha=sha(run/'best.pt')
        else:
            assert str(d['dates'][0])=='2022-05-26'
            model_sha=sha(ROOT/'results/sp500-no_skip-seed0/best.pt')
            assert model_sha==sha(ROOT/'results/archive/long_only_2026-11-05/sp500-no_skip-seed0/best.pt')
        checks.append({'run':run.name,'daily_neutrality':'pass','buy_only_dollar_ledger':'pass',
                       'metric_recomputation':'pass','time_boundaries':'pass','checkpoint_sha256':model_sha})
    (root/'verification.json').write_text(json.dumps({'source_hashes':'pass','protocol':PORTFOLIO_PROTOCOL,'runs':checks},indent=2)+'\n')
    print(f'PASS: all 10 recent tests; source hashes, chronological selection, neutral books, and independent buy-only cash ledger.')


if __name__=='__main__':main()

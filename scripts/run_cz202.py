"""Frozen monthly CZ202 adaptation; preserve the original daily experiments."""
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transalpha.data import save_json, sha
from transalpha.model import TransAlpha, LinearControl
from transalpha.loss import hybrid_loss
from transalpha.evaluation import correlation

BASE = Path('/home/jkp/data/.replication-local/artifacts/ordered-reproductions')
OUT = ROOT / 'results/cz202'


class MonthlyPanel:
    def __init__(self):
        source = BASE / 'data/cz-monthly/panel.parquet'
        self.manifest = json.loads((source.parent / 'manifest.json').read_text())
        assert sha(source) == self.manifest['panel_sha256']
        self.names = self.manifest['features']
        assert len(self.names) == 202
        self.frame = pd.read_parquet(source).sort_values(['return_month', 'permno']).reset_index(drop=True)
        f = self.frame
        assert not f.duplicated(['return_month', 'permno']).any()
        assert (f.lagged_market_cap >= f.nyse20).all()
        periods = pd.PeriodIndex(pd.to_datetime(f.return_month.astype(str), format='%Y%m'), freq='M')
        self.months = sorted(f.return_month.unique())
        assert len(pd.period_range(periods.min(), periods.max(), freq='M')) == len(self.months)
        symbols = sorted(f.permno.unique())
        si = pd.Index(symbols).get_indexer(f.permno)
        ti = pd.Index(self.months).get_indexer(f.return_month)
        self.lookup = np.zeros((len(self.months), len(symbols)), dtype=np.int32)
        self.lookup[ti, si] = np.arange(len(f)) + 1
        self.x = np.vstack([np.zeros((1, 202), np.float32), f[self.names].to_numpy(np.float32)])
        assert np.isfinite(self.x).all()
        self.y = f.target_return.to_numpy(np.float32)
        assert np.isfinite(self.y[f.return_month >= 200901]).all()
        self.rows = [np.flatnonzero(ti == t) for t in range(len(self.months))]
        self.stocks = [si[r] for r in self.rows]
        self.splits = {k: [i for i,m in enumerate(self.months) if a <= m <= b]
                       for k,(a,b) in dict(train=(200901,201912),validation=(202001,202112),test=(202201,202512)).items()}
        missing = {}
        for split, ids in self.splits.items():
            slots = absent = 0
            for t in ids:
                lookup = self.history_ids(t)
                slots += lookup.size
                absent += int((lookup == 0).sum())
            rows = np.concatenate([self.rows[t] for t in ids])
            missing[split] = dict(months=len(ids),rows=len(rows),history_stock_month_slots=slots,
                absent_history_stock_months=absent,missing_characteristic_cells=int((202-f.iloc[rows].observed_feature_count).sum()),
                total_characteristic_cells=int(len(rows)*202),missing_targets=int((~np.isfinite(self.y[rows])).sum()))
        save_json(OUT / 'data_audit.json', dict(passed=True,source_sha256=sha(source),features=self.names,
            missingness=missing,calendar_contiguous=True,unique_stock_months=True,nyse20_screen=True,
            history='Calendar-aligned 16 months; absent stock-months neutral zero; no forward/back filling',
            source_manifest=self.manifest))

    def history_ids(self, t):
        assert t >= 15
        return self.lookup[t-15:t+1, self.stocks[t]].T

    def batch(self,t):
        return torch.from_numpy(self.x[self.history_ids(t)]), torch.from_numpy(self.y[self.rows[t]])


def predictions(model, panel, ids):
    model.eval()
    scores, ics, rics = [], [], []
    with torch.no_grad():
        for t in ids:
            x,y = panel.batch(t)
            p = model(x).numpy()
            assert np.isfinite(p).all()
            scores.append(p)
            ics.append(correlation(p,y.numpy()))
            rics.append(float(spearmanr(p,y.numpy()).statistic))
    return scores, ics, rics


def fit(panel, variant, seed, protocol, evaluator):
    out = OUT / f'{variant}-seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        assert json.loads((out/'metrics.json').read_text())['protocol_sha256'] == sha(OUT/'protocol.json')
        print('Verified completed run: '+str(out),flush=True)
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = TransAlpha(202,64,.1,.001) if variant == 'transalpha' else LinearControl(202)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    rng = np.random.default_rng(seed)
    best,bad = -float('inf'),0
    history=[]; started=time.monotonic()
    for epoch in range(1,31):
        model.train(); losses=[]
        for g in optimizer.param_groups:g['lr']=.001/np.sqrt(epoch)
        for t in rng.permutation(panel.splits['train']):
            x,y=panel.batch(t)
            optimizer.zero_grad(set_to_none=True)
            loss,_=hybrid_loss(model(x),y,model.parameters())
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        _,ic,_=predictions(model,panel,panel.splits['validation'])
        score=float(np.mean(ic))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_ic=score,seconds=time.monotonic()-started)
        history.append(row);save_json(out/'history.json',history)
        print(json.dumps(dict(variant=variant,seed=seed,**row)),flush=True)
        if score>best+1e-6:
            best,bad=score,0
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,protocol=protocol),out/'model.pt')
        else:bad+=1
        if bad>=5:break
    ckpt=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    scores,ics,rics=predictions(model,panel,panel.splits['test'])
    rows=np.concatenate([panel.rows[t] for t in panel.splits['test']])
    test=panel.frame.iloc[rows].copy()
    exported=test[['return_month','permno']].copy();exported['score']=np.concatenate(scores)
    exported.to_csv(out/'scores.csv',index=False)
    monthly,holdings=evaluator.portfolios(test,exported,10)
    factors=pd.read_csv(BASE/'data/factors-monthly/ff5-umd.csv')
    regression=evaluator.regress(monthly,factors,6)
    monthly['ic']=ics;monthly['rank_ic']=rics
    monthly.to_csv(out/'monthly_returns.csv',index=False)
    holdings.to_parquet(out/'holdings.parquet',index=False)
    # Independently reconstruct value-weighted decile returns from saved holdings.
    replay=holdings.merge(test[['return_month','permno','target_return']],on=['return_month','permno'],validate='one_to_one')
    gross=(replay.ls_weight*replay.target_return).groupby(replay.return_month).sum().to_numpy()
    np.testing.assert_allclose(gross,monthly.gross,rtol=0,atol=1e-12)
    np.testing.assert_allclose(monthly.net,monthly.gross-.001*(monthly.turnover+monthly.terminal_turnover),atol=1e-12)
    # Load the checkpoint into a separate instance and replay held-out months.
    fresh=TransAlpha(202,64,.1,.001) if variant=='transalpha' else LinearControl(202)
    fresh.load_state_dict(ckpt['state_dict'])
    check,_,_=predictions(fresh,panel,[panel.splits['test'][i] for i in [0,24,47]])
    for a,i in zip(check,[0,24,47]):np.testing.assert_array_equal(a,scores[i])
    metrics=dict(scope=protocol['scope'],variant=variant,seed=seed,test_months=48,
        ic=float(np.mean(ics)),rank_ic=float(np.mean(rics)),best_epoch=ckpt['epoch'],epochs=len(history),
        validation_ic=best,seconds=time.monotonic()-started,regression=regression,
        gross_sharpe=float(monthly.gross.mean()/monthly.gross.std(ddof=1)*np.sqrt(12)),
        net_sharpe=float(monthly.net.mean()/monthly.net.std(ddof=1)*np.sqrt(12)),
        protocol_sha256=sha(OUT/'protocol.json'),checkpoint_sha256=sha(out/'model.pt'),scores_sha256=sha(out/'scores.csv'),
        checks=dict(passed=True,checkpoint_replay=True,decile_return_replay=True,fee_identity=True,full_test_universe=True))
    save_json(out/'metrics.json',metrics)
    print(json.dumps(dict(completed=str(out),ic=metrics['ic'],net_sharpe=metrics['net_sharpe'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',default='0,1,2');parser.add_argument('--variants',default='transalpha,linear')
    args=parser.parse_args();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    evaluator_path=BASE/'scripts/evaluate_cz.py'
    spec=importlib.util.spec_from_file_location('cz_evaluation',evaluator_path);evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    protocol=dict(scope='Monthly US CZ202 adaptation, not the private 15-minute CSI paper benchmark',
        features=202,lookback_months=16,hidden=64,train=[200901,201912],validation=[202001,202112],test=[202201,202512],
        target='Raw next-month CRSP total return; no target normalization',
        history_missing='Neutral-zero calendar slots for months absent from the screened source panel',
        feature_missing='Source within-month percentile rank normalization; missing characteristic=0',
        optimizer='Adam lr .001/sqrt(epoch); original five-term loss; one full monthly cross section per optimizer step',
        epochs=30,patience=5,checkpoint='Highest validation mean Pearson IC; minimum improvement 1e-6',
        universe='All source-eligible US common stocks; lagged market cap >= lagged NYSE20; no future-return filtering',
        portfolio='Value-weighted D10-D1, +1/-1 exposure; 10bp per traded dollar including opening and final closing',
        inference='FF5+UMD OLS; Bartlett HAC lags6, corrected covariance, Student t; annual alpha and CI = monthly *12',
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'transalpha/model.py',ROOT/'transalpha/loss.py']},
        evaluator_sha256=sha(evaluator_path),factor_sha256=sha(BASE/'data/factors-monthly/ff5-umd.csv'),
        torch_version=torch.__version__,numpy_version=np.__version__)
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save_json(OUT/'protocol.json',protocol)
    panel=MonthlyPanel()
    for variant in args.variants.split(','):
        assert variant in ['transalpha','linear']
        for seed in map(int,args.seeds.split(',')):fit(panel,variant,seed,protocol,evaluator)
    save_json(OUT/'completion.json',dict(complete=True,seeds=args.seeds,variants=args.variants))

if __name__=='__main__':main()

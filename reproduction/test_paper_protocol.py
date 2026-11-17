"""Regression tests for the paper-equation revision and epoch recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from backbones import make_forecaster
from build_tensors import FEATURES, cumulative_labels, features_from_grid
from core import (bilevel_step, entropy, lookahead, second_order_context,
                  supervised_step, target_loss, warmup_label, weighted_loss)
import train


class TinyStore:
    device='cpu'
    def __init__(self):
        self.splits={'train':list(range(40)),'validation':[40,41],'test':[42]}
        g=torch.Generator().manual_seed(10)
        self.items={i:(torch.randn(9,3,3,generator=g,dtype=torch.double),
                       torch.randn(9,390,generator=g,dtype=torch.double)) for i in range(43)}
        self.reads=[]
    def day(self,date):
        self.reads.append(date);return self.items[date]
    def days(self,dates):return [self.day(d) for d in dates]


class PaperProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(13)
        self.config=json.loads(Path('protocol.json').read_text())
        self.config.update(input_features=3,maximum_epochs=4,patience=5)
        self.config['model'].update(hidden_size=3,input_features=3)

    def test_patch_locality_and_no_extra_day_features(self):
        rng=np.random.default_rng(10);x=np.exp(rng.normal(size=(2,390,7)))
        before=features_from_grid(x);x[:,15:]*=7
        after=features_from_grid(x)
        np.testing.assert_array_equal(before[:,0],after[:,0])
        self.assertEqual(before.shape,(2,26,66))
        self.assertEqual(len(FEATURES),66)

    def test_independent_raw_patch_descriptors(self):
        rng=np.random.default_rng(15);x=np.exp(rng.normal(size=(1,390,7)))
        raw=x[0,:15];expected=[]
        for v in raw.T:
            z=(v-v.mean())/v.std()
            expected.extend([v.mean(),v[-1]/v[0]-1,np.ptp(v)/abs(v.mean()),v.std(),(z**3).mean(),(z**4).mean()-3])
        expected.extend(np.corrcoef(raw,rowvar=False)[i,j] for i in range(7) for j in range(i+1,7))
        amount,volume,count=raw[:,4:].sum(0)
        expected.extend([amount/volume/raw[:,3].mean()-1,volume/count,amount/count])
        np.testing.assert_allclose(features_from_grid(x)[0,0],expected,atol=1e-12,rtol=1e-12)

    def test_dense_labels_causal_and_split_adjusted(self):
        future=np.array([[np.nan,51,np.nan,55],[100,101,102,np.nan]])
        labels,keep,missing=cumulative_labels(np.array([100.,100.]),future,np.array([2.,1.]),np.ones(2))
        np.testing.assert_allclose(labels[0],[0,.02,.02,.1])
        self.assertEqual(keep.tolist(),[True,False])
        self.assertEqual(missing[0].tolist(),[True,False,True,False])
        future[0,-1]=10000
        altered,_,_=cumulative_labels(np.array([100.,100.]),future,np.array([2.,1.]),np.ones(2))
        np.testing.assert_array_equal(altered[:,:-1],labels[:,:-1])

    def test_warmup_rejects_incomplete_path(self):
        y=torch.randn(12,4,dtype=torch.double);y[0,0]=float('nan')
        with self.assertRaises(ValueError):warmup_label(y)

    def test_date_sums_are_not_averages(self):
        model=make_forecaster(self.config).double()
        days=[TinyStore().day(i) for i in range(3)]
        weights=torch.ones(390,dtype=torch.double)/390
        got=weighted_loss(model,days,weights)
        expected=sum(weighted_loss(model,[d],weights) for d in days)
        torch.testing.assert_close(got,expected)
        torch.testing.assert_close(target_loss(model,days,389),sum(target_loss(model,[d],389) for d in days))

    def test_streamed_adamw_is_one_full_batch_step(self):
        for name in ['LSTM','GRU','TCN']:
            c=copy.deepcopy(self.config);c['model']['name']=name
            a=train.new_model(c,1,'cpu');b=train.new_model(c,1,'cpu')
            oa=train.adamw(a,c);ob=train.adamw(b,c)
            days=TinyStore().days([0,1,2])
            for _ in range(3):
                oa.zero_grad();loss=target_loss(a,days,389);loss.backward();oa.step()
                observed=supervised_step(b,days,ob,389)
                self.assertAlmostEqual(observed,float(loss),places=11)
                for pa,pb in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(pa,pb,atol=1e-9,rtol=1e-8)

    def test_literal_sgd_and_student_writeback(self):
        model=train.new_model(self.config,0,'cpu');store=TinyStore()
        support,query=store.days([0,1]),store.days([2,3])
        logits=nn.Parameter(torch.randn(390,dtype=torch.double)*.01)
        with second_order_context():
            ahead,_=lookahead(model,support,logits,1e-6)
            outer=target_loss(model,query,389,params=ahead)-.001*entropy(logits)
            g,=torch.autograd.grad(outer,logits)
        expected=logits.detach()-.001*g;state={k:v.detach().clone() for k,v in ahead.items()}
        bilevel_step(model,support,query,logits,torch.optim.SGD([logits],lr=.001),1e-6,.001,389)
        torch.testing.assert_close(logits,expected)
        for name,p in model.named_parameters():torch.testing.assert_close(p,state[name])
        self.assertTrue(all(p.dtype==torch.float64 for p in model.parameters()))

    def test_supervised_epoch_recovery_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'whole.pt',Path(tmp)/'resume.pt'
            expected=train.train_supervised(TinyStore(),self.config,1,7,a,'test',lambda x:None)
            def interrupt(value):
                if value.get('epoch')==2:raise InterruptedError('fixture interruption')
            with self.assertRaises(InterruptedError):
                train.train_supervised(TinyStore(),self.config,1,7,b,'test',interrupt)
            store=TinyStore();actual=train.train_supervised(store,self.config,1,7,b,'test',lambda x:None)
            self.assertNotIn(42,store.reads)
            self.assertEqual(actual['history'],expected['history'])
            self.assertEqual(actual['validation_horizon'],7)
            for key,v in actual['state_dict'].items():torch.testing.assert_close(v,expected['state_dict'][key],rtol=0,atol=0)

    def test_bilevel_epoch_recovery_preserves_rng_logits_and_student(self):
        config=copy.deepcopy(self.config);config.update(maximum_epochs=2,warmup_epochs=3)
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'whole.pt',Path(tmp)/'resume.pt'
            expected=train.select_horizon(TinyStore(),config,1,a,'test',lambda x:None)
            def interrupt(value):
                if value.get('phase')=='bilevel':raise InterruptedError('fixture interruption')
            with self.assertRaises(InterruptedError):train.select_horizon(TinyStore(),config,1,b,'test',interrupt)
            actual=train.select_horizon(TinyStore(),config,1,b,'test',lambda x:None)
            self.assertEqual(actual['history'],expected['history'])
            self.assertEqual(actual['weights'],expected['weights'])
            for key,v in actual['state_dict'].items():torch.testing.assert_close(v,expected['state_dict'][key],rtol=0,atol=0)

    def test_no_cap_and_exact_patience_restore(self):
        config=copy.deepcopy(self.config);config['maximum_epochs']=None
        values=iter([2.,1.,1.1,1.2,1.3,1.4,1.5])
        with tempfile.TemporaryDirectory() as tmp,patch.object(train,'validation_loss',side_effect=lambda *a:next(values)):
            fit=train.train_supervised(TinyStore(),config,1,390,Path(tmp)/'fit.pt','test',lambda x:None)
        self.assertEqual(len(fit['history']),7)
        self.assertEqual(fit['best_validation_loss'],1.)
        self.assertEqual(fit['stopping_reason'],'patience')

if __name__=='__main__':unittest.main(verbosity=2)

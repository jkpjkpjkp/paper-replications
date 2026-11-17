"""Independent chronology, accounting, and learned-model checks."""
import unittest
import numpy as np
import pandas as pd
import torch
from diffusion_model import ConditionalUNet,cosine_schedule
from run_empirical import features,simulate,origins,repair_weights,ROOT,P,sampled_dtw,dtw


class EmpiricalChecks(unittest.TestCase):
    def test_buy_hold_matches_fixed_share_value(self):
        simple=np.array([[.1,-.2],[.3,.1],[-.1,.2]])
        w=np.array([1.4,-.4])
        ledger=simulate(np.log1p(simple),pd.date_range('2025-01-01',periods=3),{0:w})
        expected=np.cumprod(1+simple,axis=0)@w
        np.testing.assert_allclose(ledger.wealth,expected,atol=1e-12)

    def test_cost_charged_only_at_rebalance(self):
        ledger=simulate(np.zeros((3,2)),pd.date_range('2025-01-01',periods=3),
                        {0:np.array([1.,0.]),2:np.array([0.,1.])},cost=.001)
        np.testing.assert_allclose(ledger.turnover,[1,0,2])
        self.assertAlmostEqual(ledger.wealth.iloc[-1],.999*.998)

    def test_future_test_data_cannot_change_earlier_features(self):
        p=pd.read_parquet(ROOT/'data/US_prices.parquet')
        m=pd.read_parquet(ROOT/'data/global_macro.parquet')
        before=features(p,m)
        change=pd.Timestamp('2024-11-01')
        p2=p.copy();p2.loc[change:]*=2
        m2=m.copy();m2.loc[change:]*=3
        after=features(p2,m2)
        prefix=p.index<change
        np.testing.assert_array_equal(before['macro'][prefix],after['macro'][prefix])
        np.testing.assert_array_equal(before['stock'][prefix],after['stock'][prefix])
        np.testing.assert_array_equal(before['scale'],after['scale'])

    def test_targets_stay_in_assigned_split(self):
        for market in P['markets']:
            dates=pd.read_parquet(ROOT/f'data/{market}_prices.parquet').index
            for start,end in [(P['train_start'],P['fit_end']),(P['validation_start'],P['validation_end'])]:
                idx=origins(dates,start,end)
                self.assertTrue(all(dates[j]>=pd.Timestamp(start) and dates[j+19]<=pd.Timestamp(end) for j in idx))
                self.assertTrue(all(dates[j-2]<dates[j-1]<dates[j] for j in idx))

    def test_denoiser_and_gate_receive_gradients(self):
        torch.manual_seed(7);torch.set_num_threads(2)
        model=ConditionalUNet(15,17)
        output=model(torch.randn(8,20),torch.arange(8),torch.randn(8,15),torch.randn(8,17))
        self.assertEqual(output.shape,(8,20))
        output.square().mean().backward()
        for layer in [model.macro,model.stock,model.gate,model.input]:
            self.assertGreater(float(layer.weight.grad.abs().sum()),0)

    def test_repairs_enforce_recorded_constraints(self):
        w,d=repair_weights(np.array([2.,-1.99]),np.eye(2))
        self.assertTrue(d['gross_cap_applied'])
        self.assertAlmostEqual(w.sum(),1)
        self.assertLessEqual(abs(w).sum(),2+1e-12)
        w,d=repair_weights(np.array([-1.,-2.]),np.eye(2))
        self.assertTrue(d['fallback']);np.testing.assert_allclose(w,[.5,.5])

    def test_cosine_schedule_approaches_pure_noise(self):
        a=cosine_schedule()
        self.assertEqual(len(a),200)
        self.assertTrue(torch.all(a[1:]<a[:-1]))
        self.assertLess(float(a[-1]),1e-6)

    def test_sampled_dtw_matches_individual_path_reference(self):
        rng=np.random.default_rng(12)
        paths=rng.normal(size=(3,5,2));truth=rng.normal(size=(5,2))
        expected=np.mean([dtw(paths[k,:,s],truth[:,s]) for k in range(3) for s in range(2)])
        self.assertAlmostEqual(sampled_dtw(paths,truth),expected,places=12)


if __name__=='__main__':unittest.main(verbosity=2)

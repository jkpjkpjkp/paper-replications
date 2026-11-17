"""Protect validation-only selection and full-grid completeness."""
import unittest
from controller import select_configs


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.protocol={'asset_pools':[40], 'configurations':[{'d_model':8},{'d_model':16}],
                       'seeds':[2023,2024,2025,2026,2027]}
        self.results={}
        for i in range(2):
            for seed in self.protocol['seeds']:
                self.results[(40,i,seed)]=dict(status='completed',test_data_loaded=False,
                    pool=40,config_index=i,seed=seed,best_validation_cvar=float(2-i),
                    deliberately_misleading_test_sharpe=float(10-10*i))

    def test_validation_only_winner(self):
        self.assertEqual(select_configs(self.protocol,self.results)['40']['config_index'],1)

    def test_missing_seed_rejects_selection(self):
        del self.results[(40,0,2027)]
        with self.assertRaises(KeyError):
            select_configs(self.protocol,self.results)

    def test_test_access_rejects_selection(self):
        self.results[(40,0,2027)]['test_data_loaded']=True
        with self.assertRaises(AssertionError):
            select_configs(self.protocol,self.results)

    def test_tie_uses_preregistered_index(self):
        for result in self.results.values():
            result['best_validation_cvar']=1.0
        self.assertEqual(select_configs(self.protocol,self.results)['40']['config_index'],0)


if __name__=='__main__':
    unittest.main()

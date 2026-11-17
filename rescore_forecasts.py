"""Correct sampled-path DTW from frozen predictions; no retraining or selection."""
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from run_empirical import ROOT,OUT,P,sampled_dtw


def main():
    for m in P['markets']:
        prices=pd.read_parquet(ROOT/f'data/{m}_prices.parquet')
        r=np.log(prices).diff()
        for seed in P['seeds']:
            file=OUT/f'{m}_{seed}_result.json'
            result=json.loads(file.read_text())
            paths=np.load(OUT/f'{m}_{seed}_generated_paths.npz')
            score={}
            for day,a in zip(paths['first_return_dates'],paths['paths']):
                j=prices.index.get_loc(day)
                if j+20<=len(prices):
                    truth=r.iloc[j:j+20].to_numpy()
                    score[day]=(sampled_dtw(a,truth),float(np.square(a.mean(axis=0)-truth).mean()))
            for row in result['forecast_metrics']:
                if 'mean_path_normalized_DTW' not in row:
                    row['mean_path_normalized_DTW']=row['normalized_DTW']
                if row['method']=='STABLE':
                    dtw,mse=score[row['first_return_date']]
                    assert np.isclose(mse,row['MSE'],rtol=1e-12,atol=1e-15)
                    row['normalized_DTW']=dtw
            state=torch.load(ROOT/f'checkpoints/{m}_{seed}.pt',map_location='cpu',weights_only=False)
            for source in ['run_empirical.py','evidence/training_source.py']:
                fingerprint=hashlib.sha256((ROOT/'protocol.json').read_bytes()+
                    (ROOT/'diffusion_model.py').read_bytes()+(ROOT/source).read_bytes()+
                    (ROOT/f'data/{m}_prices.parquet').read_bytes()+
                    (ROOT/'data/global_macro.parquet').read_bytes()).hexdigest()
                if fingerprint==state['fingerprint']:
                    result['training_source']=source;break
            else:raise AssertionError('checkpoint source/data mismatch')
            result['forecast_scoring']='MSE of mean path; mean L1 DTW over each generated path, divided by horizon'
            file.write_text(json.dumps(result,indent=2))
    print('Rescored 12 saved forecast runs; original MSE and checkpoint provenance verified.')


if __name__=='__main__':main()

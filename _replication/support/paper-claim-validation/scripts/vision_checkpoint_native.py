#!/usr/bin/env python3
"""Regenerate observed CRSP charts and replay released image checkpoints.

Three fixed source-file dates and twelve identifier quantiles are an engineering
audit, not a paper-wide performance replication. No image or price is fabricated.
All chart drawing and network definitions are the author's unmodified code.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'paper-claim-validation'
sys.path.insert(0,str(BASE/'vision-deps'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

SOURCE=ROOT/'vision-to-value-evaluation/upstream'
OUT=BASE/'results/vision_checkpoint'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);os.chdir(OUT)
    pa.set_cpu_count(1);torch.set_num_threads(2)
    dates=sorted((SOURCE/'sdf_analysis/pred/ViT').glob('*.csv'))
    chosen=[dates[0].stem,dates[len(dates)//2].stem,dates[-1].stem]
    records=[];sources=[]
    for date in chosen:
        predictions={name:pd.read_csv(SOURCE/f'sdf_analysis/pred/{folder}/{date}.csv').set_index('ticker')
                     for name,folder in [('vit','ViT'),('cnn','cnn')]}
        ids=sorted(set(predictions['vit'].index)&set(predictions['cnn'].index))
        ids=[int(ids[i]) for i in np.linspace(0,len(ids)-1,12).astype(int)]
        end=pd.Timestamp(date);start=end-pd.Timedelta(days=120)
        parts=[]
        for year in range(start.year,end.year+1):
            path=ROOT/f'crsp-daily/parquet/{year}.parquet'
            table=pq.read_table(path,columns=['permno','dlycaldt','dlyprc','dlyprcflg',
                    'dlyopen','dlyhigh','dlylow','dlyvol','dlyret'],filters=[('permno','in',ids)])
            z=table.to_pandas();z['dlycaldt']=pd.to_datetime(z.dlycaldt)
            parts.append(z[z.dlycaldt.between(start,end)])
            sources.append(dict(path=str(path),role='Observed CRSP CIZ daily source partition',year=year))
        raw=pd.concat(parts).sort_values(['permno','dlycaldt'])
        for permno in ids:
            data=raw[raw.permno.eq(permno)].tail(60).copy().set_index('dlycaldt')
            if len(data)<44 or data.index[-1]!=end:
                records.append(dict(date=date,permno=permno,eligible=False,reason='Insufficient observed history or no exact source-date bar'))
                continue
            for column in ['dlyprc','dlyopen','dlyhigh','dlylow','dlyvol','dlyret']:
                data[column]=pd.to_numeric(data[column],errors='coerce').astype(float)
            # Legacy negative closing prices represent bid/ask averages. CIZ
            # retains a positive number with BA flag; restore that same exclusion.
            data.loc[data.dlyprcflg.eq('BA'),'dlyprc']=np.nan
            ret=data.dlyret.copy();ret.iloc[0]=np.nan
            close=(1+ret).cumprod();close.iloc[0]=1.
            p=pd.DataFrame(dict(OPEN=data.dlyopen/data.dlyprc*close,
                HIGH=data.dlyhigh/data.dlyprc*close,LOW=data.dlylow/data.dlyprc*close,
                CLOSE=close,VOL=data.dlyvol))
            tradable=data.dlyvol.ne(0).replace(False,np.nan).astype(float)
            p.loc[:,['OPEN','HIGH','LOW','CLOSE']]=p[['OPEN','HIGH','LOW','CLOSE']].mul(tradable,axis=0)
            folder=OUT/'input'/date;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{permno}.pkl';p.to_pickle(path)
            records.append(dict(date=date,permno=permno,eligible=True,input=str(path),
                history_first=str(p.index[0].date()),history_last=str(p.index[-1].date()),rows=len(p),
                expected={name:[float(predictions[name].loc[permno,f'pred{i}']) for i in range(1,6)] for name in predictions}))
    manifest=dict(scope='Predeclared three dates and twelve identifier quantiles; observed-input engineering replay',
        dates=chosen,records=records,sources=sources,
        checkpoint_sha256={str(p):sha(p) for p in sorted((SOURCE/'models/weights').glob('*/*/best_model.pth'))},
        source_sha256={str(p):sha(p) for p in [SOURCE/'models/inference.py',
            SOURCE/'image_generation/rgb_image_vit_byun.py',SOURCE/'image_generation/gray_image_cnn_xiu.py']},
        differences=['Current-vintage CIZ mapped to legacy CRSP field meanings.',
                     'Prices reconstructed over last 60 observed rows; chart scale is invariant to starting price.',
                     'No original image-training data or training run reproduced.'])
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for filename,function in [('rgb_image_vit_byun.py','pickle_to_rgbimage'),('gray_image_cnn_xiu.py','pickle_to_grayimage')]:
        module=runpy.run_path(str(SOURCE/'image_generation'/filename))
        for r in records:
            if r['eligible']:module[function](r['input'])
    module=runpy.run_path(str(SOURCE/'models/inference.py'))
    compared=[]
    for model,color in [('vit','rgb'),('cnn','gray')]:
        selected=[];images=[]
        for r in records:
            if not r['eligible']:continue
            path=OUT/f'data/image/{color}'/r['date']/f"{r['permno']}.png"
            if not path.exists():continue
            with Image.open(path) as image:
                array=np.array(image,copy=True)
            tensor=torch.from_numpy(array)
            tensor=tensor.permute(2,0,1) if tensor.ndim==3 else tensor.unsqueeze(0)
            images.append(tensor.float()/255);selected.append(r)
        assert images,f'No valid native observed charts for {model}'
        models=module['load_models'](model,str(SOURCE/f'models/weights/{model}'),'cpu')
        batch=torch.stack(images)
        for seed,net in enumerate(models,1):
            values=[]
            for start in range(0,len(batch),4):
                with torch.no_grad():values.extend(torch.softmax(net(batch[start:start+4]),dim=1)[:,1].numpy())
            for r,value in zip(selected,values):
                target=r['expected'][model][seed-1]
                compared.append(dict(model=model,date=r['date'],permno=r['permno'],seed_column=f'pred{seed}',
                    released=target,replayed=float(value),absolute_error=abs(float(value)-target)))
        del models
    frame=pd.DataFrame(compared);frame.to_csv(OUT/'checkpoint_prediction_comparison.csv',index=False)
    record=dict(status='completed_observed_input_checkpoint_probe',updated_utc=datetime.now(timezone.utc).isoformat(),
        tested_probabilities=len(frame),checkpoint_count=10,original_image_checkpoint_availability='present_and_loadable',
        within_1e_minus_4=int((frame.absolute_error<=1e-4).sum()),
        max_absolute_error=float(frame.absolute_error.max()),mean_absolute_error=float(frame.absolute_error.mean()),
        full_paper_reproduced=False,
        interpretation='A mismatch cannot isolate code error from CIZ/vintage/chart reconstruction differences. A match validates only these predeclared observed chart inputs.')
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()

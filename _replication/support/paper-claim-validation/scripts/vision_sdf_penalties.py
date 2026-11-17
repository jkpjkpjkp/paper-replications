#!/usr/bin/env python3
"""Reproduce Table 3 OLS at its published penalties while full CV continues."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from vision_sdf_native import ROOT,BASE,OUT,prepare,execute_source

prepare();os.chdir(OUT)
rows=[]
for model,coefficient,t_stat in [('vit_Cap_VW',.052,2.948),('cnn_Cap_VW',.054,4.458)]:
    env=execute_source(ROOT/'vision-to-value-evaluation/sources/sdf_loading.py',model,
                       stop_at_ols=True,published_penalties=True)
    fit=env['ols_model'];i=env['var_names'].index(model.split('_')[0])
    row=dict(model=model,first_log_lambda=float(np.log(env['lambdas1'][env['first_lambda_idx']])),
        second_log_lambda=float(np.log(env['lambdas2'][env['second_lambda_idx']])),
        sdf_loading=float(np.asarray(fit.params)[i]),sdf_loading_t=float(np.asarray(fit.tvalues)[i]),
        paper_loading=coefficient,paper_loading_t=t_stat,
        coefficient_matches_printed_precision=abs(float(np.asarray(fit.params)[i])-coefficient)<.000500001,
        t_stat_matches_printed_precision=abs(float(np.asarray(fit.tvalues)[i])-t_stat)<.000500001,
        selected_controls=env['feat_names'],portfolios=int(fit.nobs),
        scope='Published penalties supplied; their 200-seed selection is not validated by this calculation',
        full_paper_reproduced=False)
    rows.append(row)
    (OUT/(model+'_published_penalty_OLS.txt')).write_text(fit.summary().as_text()+'\n')
pd.DataFrame(rows).to_csv(OUT/'published_penalty_table3.csv',index=False)
(OUT/'published_penalty_table3.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))

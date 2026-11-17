"""Arithmetic consistency checks on visually checked transcriptions of Tables7-8."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    t7={
      'w/o Graph':[.0675,.1728,.0896,.2135,.1166,.3625,.0912,.2496],
      'w/o News':[.0745,.1741,.0903,.2147,.1287,.3985,.0978,.2624],
      'w/o QI':[.0777,.1703,.0920,.2157,.1478,.4168,.1058,.2676],
      'w/o HGA':[.0754,.1759,.0936,.2178,.1552,.4274,.1081,.2737],
      'w/o MT':[.0781,.1754,.0929,.2228,.1577,.4485,.1096,.2822],
      'Ours':[.0859,.1820,.1006,.2252,.1681,.4587,.1182,.2886]}
    cols=['prec3','mrr3','prec5','mrr5','prec10','mrr10','avg_prec','avg_mrr']
    pd.DataFrame.from_dict(t7,orient='index',columns=cols).rename_axis('model').to_csv(ROOT/'results/published_table7.csv')
    t8={
      'w/o Graph':[[.4365,1.6565,.1327],[.3429,1.5336,.1601],[.2529,1.2987,.1274],[.3441,.3441,.1401]],
      'w/o News':[[.4428,1.8482,.0843],[.3436,1.6483,.0816],[.2600,1.5028,.0650],[.3488,.3488,.0770]],
      'w/o QI':[[.4455,1.9956,.0801],[.3430,1.7174,.0876],[.2667,1.5052,.0595],[.3517,.3517,.0757]],
      'w/o HGA':[[.4687,2.0126,.0907],[.3437,1.8973,.0720],[.2742,1.5971,.0668],[.3622,.3622,.0765]],
      'w/o MT':[[.4716,2.1384,.1101],[.3471,1.8592,.0822],[.2720,1.6135,.0900],[.3636,.3636,.0941]],
      'w/o Risk':[[.5071,2.2669,.1360],[.3792,1.8668,.1141],[.2824,1.6446,.1173],[.3896,.3896,.1225]],
      'Ours':[[.5123,2.4092,.0792],[.3817,1.9511,.0806],[.2828,1.7400,.0664],[.3923,.3923,.0754]]}
    rows=[];bounds=[];avg=[]
    for model,values in t7.items():
        for k,pos in [(3,1),(5,3),(10,5)]:
            upper=float(np.sum(1/np.arange(1,k+1))/k)
            bounds.append({'model':model,'k':k,'reported_mrr':values[pos],'eq20_max':upper,'violates':values[pos]>upper+.00005})
    for model,values in t8.items():
        for k,v in zip([3,5,10,'Avg'],values):rows.append({'model':model,'k':k,**dict(zip(['irr','sharpe','mdd'],v))})
        actual=np.mean(values[:3],axis=0)
        for i,m in enumerate(['irr','sharpe','mdd']):
            avg.append({'model':model,'metric':m,'reported_avg':values[3][i],'recomputed_from_rows':float(actual[i]),'consistent_at_printed_precision':abs(actual[i]-values[3][i])<=.0001})
    pd.DataFrame(rows).to_csv(ROOT/'results/published_table8.csv',index=False)
    pd.DataFrame(bounds).to_csv(ROOT/'results/mrr_bounds.csv',index=False)
    pd.DataFrame(avg).to_csv(ROOT/'results/table8_average_checks.csv',index=False)
    result={'source':'sources/paper.pdf; visually checked PDF pages 9,10,12',
            'pdf_sha256':hashlib.sha256((ROOT/'sources/paper.pdf').read_bytes()).hexdigest(),
            'mrr_equation20':'mean reciprocal true ranking position among K distinct predicted stocks',
            'eq20_top10_upper_bound':float(np.sum(1/np.arange(1,11))/10),
            'table7_mrr_bound_violations':sum(x['violates'] for x in bounds),
            'table8_average_sharpe_errors':sum(not x['consistent_at_printed_precision'] for x in avg if x['metric']=='sharpe'),
            'interpretation':'Tables cannot all be reproduced while obeying the printed definitions. Metric definition or table corrections are required; this does not identify the authors actual implementation.',
            'other_ambiguities':['Eq14 prints no squared deviations, conflicting with prose standard deviation',
                                  'Eq23 mixes daily mean returns and annual risk-free rate without explaining annualization',
                                  'walk-forward refit cadence/window/update details absent',
                                  'head widths/dropout rate/early-stop patience unspecified',
                                  'point-in-time date of graph and MSC pretraining split/checkpoint absent'],
            'exact_replication_status':'blocked_missing_original_inputs_and_definition_inconsistencies',
            'independent_full_replication_found':False,
            'search_scope':'Saved Google exact-title/model/author searches plus GitHub repository searches; no exhaustive claim about all private/public work',
            'missing_original_inputs':['RESSET 39-column panel','Hundsun news and trained MSC features','336 Qichacha typed edges','author source/checkpoints','daily predictions and training seeds'],
            'full_experiment_coverage':{'Table5_baselines':'not reproduced','Table6_baseline_portfolios':'not reproduced','Table7_exact_ablation':'blocked','Table8_exact_ablation':'blocked','Section5.3_sweeps':'not reproduced','AppendixC_Top1':'not reproduced'}}
    (ROOT/'results/paper_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

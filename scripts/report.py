#!/usr/bin/env python3
"""Build tables, scientific figures and an HTML execution report from saved results."""
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from krrdp.market import Market
from krrdp.reference import geometric_tree

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                     'figure.facecolor':'white','axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
                     'savefig.dpi':170})
BLUE, ORANGE, GREEN = '#175d8d', '#d77428', '#168174'
PAPER = json.loads((ROOT/'sources/paper_results.json').read_text())


def summarize(directory):
    rows=[]
    paths=sorted((ROOT/directory/'runs').glob('*.json'))
    groups={}
    for path in paths:
        row=json.loads(path.read_text())
        groups.setdefault((row['kind'], row['market']['dimension']), []).append(row)
    refs=json.loads((ROOT/directory/'references.json').read_text())
    for (kind,dimension),runs in sorted(groups.items()):
        values=np.array([r['price'] for r in runs])
        mean=float(values.mean())
        std=float(values.std(ddof=1))
        half=float(student_t.ppf(.975,len(values)-1)*std/np.sqrt(len(values)))
        paper=next(x for x in PAPER[kind] if x['dimension']==dimension)
        reference=refs[str(dimension)][kind]
        rows.append(dict(kind=kind, dimension=dimension, repetitions=len(values),
             mean=mean,std=std,ci_low=mean-half,ci_high=mean+half,
             mean_seconds=float(np.mean([r['seconds'] for r in runs])),
             paper_price=paper['price'],paper_ci_low=paper['ci_low'],paper_ci_high=paper['ci_high'],
             paper_seconds=paper['seconds'],reference=reference,
             difference_to_paper=mean-paper['price'],difference_to_reference=mean-reference,
             relative_error_reference=abs(mean-reference)/reference,
             mean_inside_paper_ci=bool(paper['ci_low']<=mean<=paper['ci_high']),
             ci_halfwidth_ratio=half/((paper['ci_high']-paper['ci_low'])/2),
             n=runs[0]['training']['n'], m=runs[0]['training']['m']))
    return rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def save(figure,name):
    figure.savefig(FIG/(name+'.png'),bbox_inches='tight')
    figure.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(figure)


def main():
    # Fail closed rather than publishing a report for a partially completed run.
    for directory in ['main','ten_steps','independent_mc','convergence']:
        if not (ROOT/'results'/directory/'COMPLETE.json').exists():
            raise RuntimeError(f'Incomplete experiment: {directory}')
    summaries={name:summarize('results/'+name) for name in ['main','ten_steps','independent_mc']}
    rows=summaries['main']
    for name,data in summaries.items():
        write_csv(ROOT/'results'/name/'summary.csv',data)
    # Table replication: show confidence widths as well as means.
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,kind,title in zip(axes,['geometric_put','max_call'],['Geometric basket put','Maximum call']):
        data=[r for r in rows if r['kind']==kind]
        x=np.arange(len(data))
        ax.errorbar(x-.10,[r['mean'] for r in data],
                    yerr=[r['ci_high']-r['mean'] for r in data],fmt='o',capsize=4,
                    color=BLUE,label='Reconstruction: mean + 95% t CI')
        ax.errorbar(x+.10,[r['paper_price'] for r in data],
                    yerr=[[r['paper_price']-r['paper_ci_low'] for r in data],
                          [r['paper_ci_high']-r['paper_price'] for r in data]],
                    fmt='s',capsize=3,color=ORANGE,label='Paper: reported mean + 95% CI')
        ax.plot(x,[r['reference'] for r in data],'k_',markersize=13,label='Independent same-grid reference')
        ax.set_xticks(x,[str(r['dimension']) for r in data]);ax.set_xlabel('Number of assets d')
        ax.set_ylabel('Option value');ax.set_title(title)
    axes[0].legend(loc='best',fontsize=8)
    fig.suptitle('Tables 1 and 2: independent KRR-DP reconstruction (10 seeds)',fontsize=14)
    save(fig,'table_comparison')

    # Convergence at the fixed initial spot, i.e. the mu_0 point mass.
    cr=[]
    for path in sorted((ROOT/'results/convergence').glob('d*.json')):
        r=json.loads(path.read_text());cr.append(r)
    conv=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,dim in zip(axes,[20,40]):
        xs=sorted(set(r['training']['n'] for r in cr if r['market']['dimension']==dim))
        means=[];ses=[]
        for n in xs:
            batch=[r for r in cr if r['market']['dimension']==dim and r['training']['n']==n]
            es=np.array([r['squared_error'] for r in batch])
            mean=float(es.mean());se=float(es.std(ddof=1)/np.sqrt(len(es)))
            means.append(mean);ses.append(se)
            conv.append(dict(dimension=dim,n=n,m=batch[0]['training']['m'],
                             repetitions=len(batch),mse=mean,se=se,
                             reference=batch[0]['reference'],max_centers=256))
        ax.errorbar(xs,means,yerr=ses,fmt='o-',capsize=4,color=BLUE)
        ax.set_xlabel('Training states per stage n');ax.set_ylabel('Mean squared price error')
        ax.set_title(f'd = {dim} | 5 seeds | mean + 1 SE');ax.set_ylim(bottom=0)
    fig.suptitle('Figure 2 reconstruction: direct Nyström (up to 256 centers)',fontsize=13)
    save(fig,'convergence')
    write_csv(ROOT/'results/convergence/summary.csv',conv)

    # Exact decomposition and MC scaling diagnostic.
    diagnostic=json.loads((ROOT/'results/diagnostics.json').read_text())
    batch=[r for r in diagnostic['decomposition'] if r['n']==128 and r['m']==32]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for key,label,marker in [('regression_mse','I: regression','o'),('mc_mse','II: Monte Carlo','s'),
                             ('propagation_mse','III: propagation','^'),('total_mse','Total error','D')]:
        values=[np.mean([next(s[key] for s in r['stages'] if s['stage']==stage) for r in batch])
                for stage in range(9)]
        values=np.where(np.asarray(values)>0,values,np.nan)
        axes[0].semilogy(range(9),values,marker=marker,label=label)
    axes[0].set_xlabel('Stage t (recursion runs right to left)');axes[0].set_ylabel('Squared L2 error under exact mu_t')
    axes[0].set_title('Exact finite-state diagnostic | 20 seeds')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,fontsize=8)
    mc=diagnostic['mc_scaling']
    axes[1].loglog([r['m'] for r in mc],[r['mse'] for r in mc],'o-',color=BLUE,label='Measured (2000 repetitions)')
    axes[1].loglog([r['m'] for r in mc],[r['exact_mse'] for r in mc],'--',color=ORANGE,label='Exact Var(f) / M')
    axes[1].set_xlabel('Monte Carlo samples M');axes[1].set_ylabel('Conditional-expectation MSE')
    axes[1].set_title('Isolating the 1/M sampling error');axes[1].legend(fontsize=8)
    save(fig,'error_diagnostics')
    slope=float(np.polyfit(np.log([r['m'] for r in mc]),np.log([r['mse'] for r in mc]),1)[0])

    for kind,title in [('geometric_put','Geometric basket put'),('max_call','Maximum call')]:
        p=ROOT/'results/surfaces'/f'{kind}.npz'
        if not p.exists():
            raise RuntimeError('Surface experiment incomplete')
        a=np.load(p);xx,yy=np.meshgrid(a['axis'],a['axis'])
        fig=plt.figure(figsize=(12,7.4),layout='constrained')
        for i,stage in enumerate([0,2,4,6,8,9]):
            ax=fig.add_subplot(2,3,i+1,projection='3d')
            ax.plot_surface(xx,yy,a[f'value_{stage}'],cmap='viridis',rstride=2,cstride=2,
                            linewidth=.1,edgecolor='#20454a',alpha=.97)
            ax.set_xlabel('Asset 1',labelpad=1);ax.set_ylabel('Asset 2',labelpad=1)
            ax.set_zlabel('Value',labelpad=2);ax.set_title('Maturity payoff' if stage==9 else f'Stage {stage}')
            ax.tick_params(labelsize=7,pad=0);ax.view_init(elev=25,azim=-55)
        fig.suptitle(title+' | illustrative surface reconstruction\n'
                     'Initial prices sampled independently from Uniform[60, 140]; distinct from table runs',fontsize=13)
        save(fig,kind+'_surfaces')

    # Independent benchmark audit: coarse Bermudan vs near-continuous exercise.
    benchmarks=[]
    for dim in [2,5,10,20,40]:
        market=Market(dimension=dim,steps=1000)
        a=geometric_tree(market,1)
        b=geometric_tree(Market(dimension=dim,steps=2000),1)
        main_ref=next(r['reference'] for r in rows if r['kind']=='geometric_put' and r['dimension']==dim)
        paper=next(r for r in PAPER['geometric_put'] if r['dimension']==dim)
        benchmarks.append(dict(dimension=dim,bermudan_9_steps=main_ref,
                               american_1000_steps=a,american_2000_steps=b,
                               paper_benchmark=paper['benchmark']))
    write_csv(ROOT/'results/benchmark_audit.csv',benchmarks)

    def table(data,columns):
        out='<div class="table"><table><thead><tr>'+''.join('<th>'+escape(label)+'</th>' for _,label,_ in columns)+'</tr></thead><tbody>'
        for row in data:
            out+='<tr>'+''.join('<td>'+escape(formatter(row[key]))+'</td>' for key,_,formatter in columns)+'</tr>'
        return out+'</tbody></table></div>'
    fmt=lambda x:f'{x:.4f}'
    table_html=''
    for kind,title in [('geometric_put','Geometric basket put (Table 1)'),('max_call','Maximum call (Table 2)')]:
        data=[dict(r,ci=f'[{r["ci_low"]:.4f}, {r["ci_high"]:.4f}]') for r in rows if r['kind']==kind]
        table_html+='<h2>'+title+'</h2>'+table(data,[('dimension','d',str),('paper_price','Paper',fmt),
            ('mean','Reconstruction',fmt),('ci','95% t CI',str),('reference','Same-grid reference',fmt),
            ('difference_to_paper','Difference to paper',lambda x:f'{x:+.4f}'),('mean_seconds','Mean CPU seconds',lambda x:f'{x:.2f}')])
    sens=[]
    for name,data in summaries.items():
        for r in data:
            if r['dimension'] in [2,20]:
                r=dict(r);r['scenario']=name
                r['halfwidth']=(r['ci_high']-r['ci_low'])/2
                sens.append(r)
    sensitivity_table=table(sens,[('scenario','Scenario',str),('kind','Payoff',str),('dimension','d',str),
        ('mean','Mean',fmt),('halfwidth','95% CI halfwidth',fmt),('reference','Reference',fmt)])
    benchmark_table=table(benchmarks,[('dimension','d',str),('bermudan_9_steps','9-step Bermudan',fmt),
        ('american_1000_steps','1000 exercise steps',fmt),('american_2000_steps','2000 exercise steps',fmt),
        ('paper_benchmark','Paper benchmark',fmt)])
    inside=sum(r['mean_inside_paper_ci'] for r in rows)
    checks=[s for r in diagnostic['decomposition'] for s in r['stages']]
    max_residual=max(s['identity_residual'] for s in checks)
    violations=sum(s['total_mse']>s['triangle_bound']+1e-13 or s['propagation_mse']>s['contraction_bound']+1e-13 for s in checks)
    all_run_json=[json.loads(p.read_text()) for p in (ROOT/'results/main/runs').glob('*.json')]
    total_seconds=sum(r['seconds'] for r in all_run_json)
    width_median=float(np.median([r['ci_halfwidth_ratio'] for r in rows]))
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Error Propagation in Dynamic Programming - Reproduction</title>
<style>body{{font-family:system-ui,sans-serif;line-height:1.6;color:#183044;background:#f2f5f7;margin:0}}main{{max-width:1120px;margin:auto;background:white;padding:46px 5vw 70px}}h1{{font-size:2.1rem;line-height:1.2;margin:8px 0 18px}}h2{{font-size:1.3rem;margin-top:38px}}p,li{{max-width:95ch}}a{{color:#175d8d}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:.8rem;color:#526775}}.finding{{padding:18px 22px;background:#eaf2f7;border-left:4px solid #175d8d}}.small,figcaption{{font-size:.88rem;color:#526775}}img{{max-width:100%;height:auto}}figure{{margin:28px 0}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}th{{text-align:left;background:#eaf0f4}}td,th{{padding:9px 11px;border-bottom:1px solid #dce4ea;white-space:nowrap}}.table{{overflow:auto}}code,pre{{background:#f0f3f5}}pre{{padding:18px;overflow:auto}}.tag{{display:inline-block;background:#edf4ef;padding:3px 9px;font-size:.8rem}}footer{{margin-top:45px;border-top:1px solid #dce4ea;padding-top:15px}}</style>
<main><div class="eyebrow">Independent computational reproduction · 6 November 2026</div>
<h1>Error Propagation in Dynamic Programming:<br>From Stochastic Control to American Option Pricing</h1>
<p>Target: Andrea Della Vecchia and Damir Filipović, <a href="https://arxiv.org/abs/2509.20239v2">arXiv:2509.20239v2</a> (2 October 2026). This report reconstructs Algorithm 1, Tables 1–2 and the figure types from the public paper. It is an independent implementation, not author-released software.</p>
<div class="finding"><strong>Outcome: the algorithm is implemented and validated, but the reported numerical accuracy and precision are not reproduced under the documented reconstruction.</strong>
The complete main suite contains 100 fits: 10 independent seeds for each of two payoffs in dimensions 2, 5, 10, 20 and 40. {inside}/10 reconstructed means lie inside the paper's reported confidence intervals. The median ratio of reconstructed to published CI halfwidths is {width_median:.1f}. Similar means alone do not establish replication of the reported precision.</div>
<figure><img src="figures/table_comparison.png" alt="Two pricing comparisons across all five dimensions with reconstructed and reported confidence intervals and independent reference values."><figcaption>Measured run data: <a href="results/main/summary.csv">summary.csv</a>; per-seed records include stagewise lengthscales, validation errors and timings. Intervals describe variability of the mean over 10 seeds, not the accuracy of any individual price. A single seed's standard deviation is also saved in the CSV.</figcaption></figure>
{table_html}
<h2>What was implemented</h2>
<p>The code samples the uncontrolled correlated geometric Brownian motion at each stage, estimates discounted next-stage values with Monte Carlo, forms <code>max(immediate payoff, continuation)</code>, and fits that value with Gaussian kernel ridge regression. The terminal payoff is known exactly. The main estimator uses the same M innovations for all stage inputs, as written in Algorithm 1, and fresh independent training inputs and innovations at each stage. All main runs use exact full KRR with <code>(K + nλI)α = y</code>, λ = 10<sup>−6</sup>, and RBF lengthscales 40 or 80. The holdout criterion selects a lengthscale at each stage without using any reference price.</p>
<p>The paper instead uses FALKON, a Nyström-based iterative solver. Full KRR implements the statistical estimator directly but does not reproduce its numerical solver, center budget, iteration tolerance or GPU timing. Main runtime totals {total_seconds:.1f} seconds across the 100 fits; CPU timings were measured with two BLAS threads while other workspace jobs were running. They are descriptive, not a hardware speed comparison.</p>
<h2>Reported settings and necessary assumptions</h2>
<div class="table"><table><tr><th>Setting</th><th>Paper</th><th>This reconstruction</th></tr>
<tr><td>Market</td><td>S₀ = K = 100; r = .05; σ = .2; pair correlation .2</td><td>Same; zero dividends</td></tr>
<tr><td>Time grid</td><td>Discrete horizon T = 9; cited source uses maturity 1 and 10 steps</td><td>9 equal intervals over one year; 10-step sensitivity saved separately</td></tr>
<tr><td>Training distribution μₜ</td><td>General behavior-policy marginal; experiment construction unspecified</td><td>Uncontrolled GBM marginal from deterministic S₀; μ₀ is a point mass</td></tr>
<tr><td>n, M</td><td>d=2: 200,50; d=20: 800,100; others unspecified</td><td>d=2:200,50; d=5:400,50; d=10:600,100; d=20:800,100; d=40:1200,100</td></tr>
<tr><td>Kernel selection</td><td>Grid {{40,80}}; selection rule unspecified</td><td>128 held-out states per stage, squared error against the stage's empirical target</td></tr>
<tr><td>Clipping</td><td>Symmetric clipping at an unspecified B</td><td>B=100 for put; B=10000 for call predictions; no asset-state truncation</td></tr>
<tr><td>Uncertainty</td><td>10 repetitions; 95% CIs, formula and seeds unspecified</td><td>10 seeds; Student-t interval with 9 degrees of freedom</td></tr></table></div>
<p>The time step scales both drift and Brownian variance and appears in discounting. Eq. (20)'s unit-step notation cannot simply be combined with one-year benchmark prices as nine full years. The exact solution for coincident initial inputs replaces a redundant rank-one kernel solve at time zero; it still includes the factor 1/(1+λ).</p>
<h2>Independent reference checks</h2>
<p>For the geometric put, the geometric basket is exactly a scalar GBM with variance σ²[ρ+(1−ρ)/d] and effective dividend yield (σ²−σ²<sub>G</sub>)/2. A fine CRR lattice permits exercise only at the reconstruction's designated dates. The saved reference file compares 400 and 800 substeps per exercise interval. The target paper's “Benchmark” values instead agree with a much finer exercise grid:</p>
{benchmark_table}
<p>This is a distinction between a Bermudan price and a near-continuous American price. The cited <a href="https://arxiv.org/abs/1905.09474v3">Goudenège, Molent and Zanette paper</a>, §8.1.1, explicitly describes a 1000-step CRR benchmark. A general American geometric-put price is not supplied by a closed-form formula, despite the target paper's Appendix C wording.</p>
<p>For the max-call, zero dividends and nonnegative rates imply no benefit from early exercise: discounted payoff is a submartingale by convexity and the decreasing discounted strike. We therefore independently price the European payoff by one-dimensional quadrature over the maximum of the idiosyncratic normal factors, integrating out the common Gaussian factor analytically. This applies to the stated equicorrelated market, and does not imply a closed-form benchmark for arbitrary American max-call models. The d=1 limit is checked against Black–Scholes.</p>
<h2>Time-grid and sampling sensitivities</h2>
<p>Each scenario below has 10 seeds. <code>ten_steps</code> changes only the time grid. <code>independent_mc</code> uses independent innovations for each training input; this differs from Algorithm 1's shared innovations and is an exploratory alternative. At deterministic time zero the independent variant retains n independent noisy labels and their average, giving nM simulated next states at time zero; collapsing these to one label would change its variance. The shared variant has only M distinct innovations at time zero. Thus this comparison changes dependence and the effective initial budget, and cannot identify the authors' unreported implementation.</p>
{sensitivity_table}
<h2>Error propagation diagnostics</h2>
<p>A separate bounded 61-state optimal-stopping problem makes the transition matrix, forward marginals and Bellman solution available exactly. For n ∈ {{32,64,128,256}}, M ∈ {{8,32,128}} and 20 seeds, all 240 fits were evaluated at every one of 9 stages. Regression, Monte Carlo and propagation residuals sum to the total residual with maximum absolute discrepancy {max_residual:.2g}. Across {len(checks)} stage evaluations, {violations} violations were found for either the elementary squared triangle bound Eₜ ≤ 3(Iₜ+IIₜ+IIIₜ) or the propagation inequality IIIₜ ≤ exp(−2rΔt)Eₜ₊₁. The diagnostic uses matched forward marginals, which are necessary for this L² contraction check.</p>
<figure><img src="figures/error_diagnostics.png" alt="Exact finite-state squared error components by stage and Monte Carlo MSE against the exact inverse sample-size curve."><figcaption>The conditional-expectation MSE scaling experiment uses 2000 repetitions per M and gives a log–log slope of {slope:.3f}, against the exact variance/M expectation. This is a numerical diagnostic, not a proof of the paper's theorem or its asymptotic regression rate. The squared components have cross terms and must not be added as an equality of squared errors. Propagation error is exactly zero at the last preterminal stage because the terminal payoff is known; that zero is omitted from the logarithmic axis.</figcaption></figure>
<h2>Convergence reconstruction</h2>
<p>The n sweep uses 5 seeds per setting, d=20 and 40, M=round(50√(n/200)), and a direct Nyström solve with at most 256 randomly sampled centers to keep the 4000-state runs practical. This is a declared approximation to full KRR, not the paper's unspecified FALKON configuration. Error is (Ŵ₀(100,…,100)−V₀)<sup>2</sup>, which is the L²(μ₀) squared error when μ₀ is the deterministic initial-state distribution. The same seed is paired across n, while the changed array sizes mean the underlying samples are not nested.</p>
<figure><img src="figures/convergence.png" alt="Measured squared initial-price error over sample size for dimensions 20 and 40, with standard error bars."><figcaption>These points are computed from saved runs, not digitized from the paper. Five repetitions do not establish an asymptotic rate; neither a smooth monotone decrease nor the paper's plotted magnitudes are imposed.</figcaption></figure>
<h2>Value-function surface reconstructions</h2>
<p>A deterministic μ₀ does not identify a global time-zero surface. For these illustrations only, the two initial asset prices are sampled independently from Uniform[60,140], and that distribution is propagated by GBM. Each uses n=800, M=256 and the same lengthscale grid. These are qualitative reconstructions of Figures 1 and 3 under an explicit alternative training distribution, not a replication of their unreported surface protocol. Fitted values are plotted directly; no post-fit exercise envelope is imposed. The unconstrained regressor has small negative values in some near-maturity parts of this grid (minimum about −0.74 for the call and −0.32 for the put), showing that symmetric clipping alone does not enforce financial shape constraints.</p>
<figure><img src="figures/max_call_surfaces.png" alt="Six maximum-call value surfaces at stages 0, 2, 4, 6, 8 and maturity."></figure>
<figure><img src="figures/geometric_put_surfaces.png" alt="Six geometric-put value surfaces at stages 0, 2, 4, 6, 8 and maturity."><figcaption>Raw arrays and portable model weights are in <a href="results/surfaces/">results/surfaces</a>.</figcaption></figure>
<h2>Reproduce and inspect</h2>
<pre>uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python -e . --no-deps
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q
make reproduce</pre>
<p><a href="README.md">README.md</a> explains each command and output. Runs checkpoint atomically; the main suite rejects output with a different configuration hash. <a href="sources/source_manifest.json">Source manifest</a>, <a href="results/artifact_manifest.json">artifact hashes</a>, per-seed JSON, configuration files and CSV summaries preserve provenance. No market datasets or account credentials are needed.</p>
<h2>Limits of the result</h2>
<p>This reconstructs the method and measures its behavior under transparent assumptions. It cannot resolve missing authors' seeds, μₜ construction, solver controls, all sample sizes or CI procedure. Raw GBM inputs are not compactly truncated, so the main max-call experiment is not a complete implementation of the bounded-state setup used to justify every theoretical assumption. Fixed λ, small M and a capped Nyström rank in the n sweep also prevent a claim to have verified the theorem's tuned asymptotic regime. The narrower published CIs and smooth convergence curve remain unconfirmed by this implementation. No failure to reproduce here proves that the paper's results are incorrect.</p>
<footer class="small">Generated locally from saved computational results. Source [E1]: target paper, Algorithm 1, §4, Tables 1–2 and Appendix C. Source [E2]: cited benchmark, §3 and §8.1–8.1.1. Numerical evidence: main/extension run JSON, references.json, diagnostics.json and summary CSVs. This is an execution report, not a publication-ready manuscript.</footer></main></html>'''
    (ROOT/'report.html').write_text(report)
    metrics=dict(main_runs=len(all_run_json),main_mean_inside_paper_ci=inside,
                 median_ci_halfwidth_ratio=width_median,diagnostic_stages=len(checks),
                 diagnostic_violations=violations,identity_max_abs_residual=max_residual,
                 mc_loglog_slope=slope,total_main_fit_seconds=total_seconds)
    (ROOT/'results/report_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()

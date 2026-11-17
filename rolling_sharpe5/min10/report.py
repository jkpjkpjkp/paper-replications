"""Chinese minimum-ten advantage report and static scientific figures."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def pct(x):
    return f"{100*x:.2f}%"


def ptext(x):
    return "—" if pd.isna(x) else f"{x:.4f}"


def main():
    all_stats = pd.read_csv(OUT/"performance_and_advantage.csv")
    stats = all_stats[(all_stats.availability=="published_before_year") & (all_stats.scenario=="5bps") & (all_stats.hac_lags==3)].copy()
    alpha = pd.read_csv(OUT/"alpha_diagnostics.csv")
    counts = pd.read_csv(OUT/"selection_counts.csv")
    members = pd.read_csv(OUT/"selection_members.csv")
    paths = pd.read_csv(OUT/"monthly_paths.csv")
    specs = [("min10_annual_iid", "1990-2022_complete_prefix", "原库 / 1990–2022"),
             ("min10_annual_iid", "2005-2022_complete_prefix", "原库 / 2005–2022"),
             ("min10_annual_iid_no_options", "1990-2025", "无期权数据因子 / 1990–2025"),
             ("min10_annual_iid_no_options", "2005-2025", "无期权数据因子 / 2005–2025"),
             ("min10_annual_iid_no_options", "2015-2025", "无期权数据因子 / 2015–2025")]
    rows, ar, risk_rows = [], [], []
    for method, period, label in specs:
        for allocation, allocation_label in [("ew", "等权"), ("optimized", "历史均值方差权重")]:
            s = stats[(stats.method==method) & (stats.period==period) & (stats.allocation==allocation)].iloc[0]
            rows.append({"因子库 / 期间": label, "配置": allocation_label,
                "原年化均值": pct(s.base_annual_mean), "加入后": pct(s.augmented_annual_mean),
                "年化增量": pct(s.delta_annual_mean), "增量p": ptext(s.delta_mean_p),
                "原Sharpe": f"{s.base_sharpe:.3f}", "加入后Sharpe": f"{s.augmented_sharpe:.3f}",
                "Sharpe差95%区间": f"[{s.sharpe_boot_ci_low:.3f}, {s.sharpe_boot_ci_high:.3f}]",
                "平均EARC权重": pct(s.mean_earc_weight)})
            if allocation == "ew":
                risk_rows.append({"因子库 / 期间": label,
                    "原年化波动": pct(s.base_annual_vol), "加入后年化波动": pct(s.augmented_annual_vol),
                    "原最大回撤": pct(s.base_max_drawdown), "加入后最大回撤": pct(s.augmented_max_drawdown),
                    "EARC与原池相关": f"{s.base_paper_correlation:.3f}"})
        a = alpha[(alpha.method==method) & (alpha.period==period) & (alpha.availability=="published_before_year") & (alpha.scenario=="5bps") & (alpha.hac_lags==3)].iloc[0]
        ar.append({"因子库 / 期间": label,
            "控制等权池的年化alpha": pct(a.aggregate_pool_alpha_annual), "alpha p": f"{a.aggregate_pool_alpha_p:.4f}",
            "过去β调整的年化增量": pct(a.oos_full_pool_residual_annual), "增量p": f"{a.oos_full_pool_residual_p:.4f}"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"summary_chinese.csv", index=False)
    current = members[(members.target_year==2026) & (members.method=="min10_annual_iid") & (members.availability=="published_before_year")].sort_values("selection_rank")
    current.to_csv(OUT/"pool_2026_primary.csv", index=False)
    (OUT/"pool_2026_primary.json").write_text(json.dumps({"target_year": 2026,
        "training_years": [2021,2022,2023,2024,2025], "minimum_factors": 10,
        "ranking": "largest minimum signed annual IID t, pooled60 IID t tie-break",
        "factor_count": len(current), "strict_pass_count": int(current.strict_pass.sum()),
        "factors": current.factor.tolist()}, indent=2), encoding="utf8")

    # Reference the primary complete-path sensitivity in the charts; never
    # draw a wealth line by skipping missing months in the original universe.
    plot_stats = stats[(stats.method=="min10_annual_iid_no_options") & (stats.allocation=="ew")].set_index("period")
    plot_stats = plot_stats.reindex(["1990-2025", "2005-2025", "2015-2025", "2020-2025"])
    p = paths[(paths.method=="min10_annual_iid_no_options") & (paths.availability=="published_before_year") & (paths.scenario=="5bps")].sort_values("month")
    assert p.missing_factor_returns.sum()==0
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    dates = pd.PeriodIndex(p.month, freq="M").to_timestamp()
    for col, label in [("base_ew", "10-factor EW pool (gross)"), ("augmented_ew", "Pool + EARC (EARC after 5bp)")]:
        axes[0,0].plot(dates, (1+p[col]).cumprod(), label=label)
    axes[0,0].set(title="No-options sensitivity: same total factor budget", ylabel="Wealth, initial=1")
    axes[0,0].legend(fontsize=8)
    x = np.arange(len(plot_stats))
    axes[0,1].errorbar(x, plot_stats.delta_annual_mean*100,
        yerr=np.vstack([(plot_stats.delta_annual_mean-plot_stats.delta_mean_ci_low)*100,
                        (plot_stats.delta_mean_ci_high-plot_stats.delta_annual_mean)*100]), fmt="o", capsize=5)
    axes[0,1].set(title="Annual return increment: HAC(3) 95% CI", ylabel="Percentage points", xticks=x, xticklabels=plot_stats.index)
    axes[0,1].axhline(0, color="black", linewidth=.8)
    d = plot_stats.sharpe_difference
    # Draw intervals directly; percentile bootstrap need not center at estimate.
    axes[1,0].vlines(x, plot_stats.sharpe_boot_ci_low, plot_stats.sharpe_boot_ci_high)
    axes[1,0].scatter(x, d)
    axes[1,0].set(title="Sharpe increment: paired 12-month block 95% CI", ylabel="Sharpe difference", xticks=x, xticklabels=plot_stats.index)
    axes[1,0].axhline(0, color="black", linewidth=.8)
    c = current.sort_values("annual_min_t")
    axes[1,1].barh(c.factor, c.annual_min_t)
    axes[1,1].axvline(2.200985, color="grey", linestyle="--", label="Annual IID 5% two-sided threshold")
    axes[1,1].set(title="2026 primary pool: weakest annual t across 2021–2025", xlabel="Signed t-statistic")
    axes[1,1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.savefig(OUT/"advantage.png", dpi=160)
    plt.close(fig)

    def html_table(d):
        return '<div class="scroll">'+d.to_html(index=False, border=0)+'</div>'
    cdisplay=current[["selection_rank","factor","annual_min_t","pooled_t","strict_pass"]].copy()
    cdisplay.columns=["排序","因子","最弱年度t","合并60月t","是否连续五年逐年显著"]
    cdisplay["最弱年度t"]=cdisplay["最弱年度t"].map(lambda v:f"{v:.3f}")
    cdisplay["合并60月t"]=cdisplay["合并60月t"].map(lambda v:f"{v:.3f}")

    main_eq = stats[(stats.method=="min10_annual_iid") & (stats.period=="1990-2022_complete_prefix") & (stats.allocation=="ew")].iloc[0]
    full_eq = stats[(stats.method=="min10_annual_iid_no_options") & (stats.period=="1990-2025") & (stats.allocation=="ew")].iloc[0]
    intro=f"原因子库完整1990–2022样本：等权加入后的年化收益变化{pct(main_eq.delta_annual_mean)}，Sharpe从{main_eq.base_sharpe:.3f}变为{main_eq.augmented_sharpe:.3f}。剔除全部期权数据因子的1990–2025对照：年化变化{pct(full_eq.delta_annual_mean)}，Sharpe从{full_eq.base_sharpe:.3f}变为{full_eq.augmented_sharpe:.3f}。"
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>至少10因子：EARC是否有增量优势</title>
<style>body{{font-family:system-ui,'Noto Sans CJK SC',sans-serif;color:#172536;background:#fbfcfe;max-width:1400px;margin:35px auto;padding:0 24px;line-height:1.75}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #dce3ec;text-align:left}}th{{background:#edf2f8}}.scroll{{overflow-x:auto}}.lead{{background:#eaf0f8;border-left:5px solid #285f99;padding:18px}}img{{max-width:100%;height:auto}}a{{color:#185a96}}pre{{background:#edf2f8;padding:16px;overflow:auto}}</style>
<h1>每年至少10个因子：加入EARC是否有优势</h1>
<p>2026-11-03；原始收益截至2025年底。本次规则替代此前允许空池的规则，旧报告保留作记录。</p>
<div class="lead"><b>每年因子池均已补足10个，不再为空；未发现显著的正收益或Sharpe优势。</b>{intro}波动有所降低，但需要和收益下降一起看。固定历史均值方差配置在主口径中不给EARC正权重，因此组合收益保持不变。</div>
<h2>组合收益与Sharpe</h2><p>两种配置都保持相同资金预算：10个因子等权时，EARC以第11个因子加入，各占1/11。年化均值=12×月均值。收益增量p值为双侧HAC(3)；Sharpe差区间是配对12月区块bootstrap、5000次。HAC(6)/(12)和Sharpe影响函数检验另存完整表。仅EARC扣交易费，原池仍是毛收益。</p>
{html_table(table)}
<img src="advantage.png" alt="累计收益、收益增量、Sharpe差和2026因子排名">
<h2>波动与回撤</h2><p>这些是同一条实现路径的风险描述，不是对未来风险改善的显著性结论。降低波动不自动意味着提高Sharpe。</p>{html_table(pd.DataFrame(risk_rows))}
<h2>独立增量诊断</h2><p>“控制等权池的alpha”是EARC对一个池组合的事后回归，不能替代完整10因子控制。“过去β调整的增量”每年仅用前60个月回归EARC对当年入选的全部因子，冻结β应用下一年；保留截距，不减去训练alpha。它是历史β调整后的样本外剩余收益，不是严格spanning检验，也未计入β对冲费用。</p>
{html_table(pd.DataFrame(ar))}
<h2>明确的入池规则</h2><p>目标年y只读y−5到y−1的60个月：完整数据且因子发表年份&lt;y。保留前五年每一年均正向显著（双侧p&lt;0.05）的因子；不足10个时，按五个年度中最小的有符号t值从大到小补足，平手依次看合并60个月t和缩写。负向显著不算“最显著的正收益”，不根据当前年重新翻转方向。若严格通过超过10个则全部保留。本次全部年度实际均为10个。另存HAC年度排序、合并60个月排序及不限制发表年份的敏感性。</p>
<h2>2026年主因子池</h2><p>2021–2025筛选的10个因子均属于补位，不能声称它们都通过连续五年逐年显著性。</p>{html_table(cdisplay)}
<h2>数据缺口与完整历史对照</h2><p>原库2023年入选的dVolCall等期权数据因子在2–12月缺收益；名单仍保留10个，未事后删因子、补零、填收益或重分权。故原库全区间1990–2025无法无假设计算，主历史检验使用连续完整1990–2022样本。对照按SignalDoc的Cat.Data=Options预定义类别统一排除全部9个期权数据因子（包括没有缺口的同类因子），再用同样规则补足10个，覆盖完整1990–2025；这属于明确的因子库限制，不与原库结果混称。</p>
<h2>权重、成本与推断边界</h2><p>“历史均值方差权重”只是固定方法敏感性：每年用此前60个月，协方差50%向对角矩阵收缩，γ=5，最大化年化均值减γ/2倍年化方差，权重非负且和为1；加入前后使用完全相同规则。不用当年收益选参数或权重。EARC可以被赋零权重。月度固定目标权重隐含的配置再平衡费用不可得。</p>
<p>EARC沿用独立重建的Compustat RDQ公告因子，每成交1元买入或卖出收取0.0005元；gross、5bp及5bp+1%年借券费分别为独立NAV路径。Zoo只有收益，没有持仓/换手，无法称为全组合已完全扣费的可执行回测。收益及Sharpe比较是回顾性且条件于已生成策略路径；bootstrap未在每次抽样重选历史因子或重估策略。多种排名和权重是敏感性，不应从中挑最优p值宣称确认性发现。风险下降、收益上升与Sharpe改善分开判断。</p>
<h2>文件与复现</h2><p><a href="pool_2026_primary.csv">2026主名单</a> · <a href="selection_members.csv">逐年名单</a> · <a href="selection_counts.csv">逐年数量</a> · <a href="performance_and_advantage.csv">收益/Sharpe/波动/回撤完整表</a> · <a href="annual_weights.csv">年度权重</a> · <a href="alpha_diagnostics.csv">增量诊断</a> · <a href="selection_validation.json">筛选审计</a> · <a href="design.json">检验设计</a></p>
<pre>cd /home/jkp/data/earnings-timing
.venv/bin/python rolling_sharpe5/min10/select_pools.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python rolling_sharpe5/min10/analyze.py
.venv/bin/python rolling_sharpe5/min10/report.py</pre>
<p>EARC无需重算股票持仓；复用上轮已核验数据。原论文、因子库、statsmodels文档及所用统计技能来源见<a href="../../results/report.html">前次报告来源部分</a>。本轮新增验证见<code>min10/audit/</code>。</p></html>"""
    (OUT/"report.html").write_text(html,encoding="utf8")
    (HERE/"README.txt").write_text("MINIMUM TEN FACTOR POOLS\n\nOpen results/report.html.\nKeep prior strict passers, top up to10 using worst signed annual t over prior5years.\nPrimary original universe: complete1990-2022;2023 option data unavailable.\nNo-options-category sensitivity: complete1990-2025. Each pool has10 factors.\nEARC fees5bp, incumbent factors gross. Equal weight and fixed prior-data mean-variance allocation.\n\nReproduce from project root:\n.venv/bin/python rolling_sharpe5/min10/select_pools.py\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python rolling_sharpe5/min10/analyze.py\n.venv/bin/python rolling_sharpe5/min10/report.py\n",encoding="utf8")
    parent_readme=HERE.parent/"README.txt"
    old=parent_readme.read_text()
    marker="LATEST USER RULE: minimum10 factors; open min10/results/report.html.\nThe original empty-pool results below are retained as historical analysis.\n\n"
    if not old.startswith("LATEST USER RULE:"):
        parent_readme.write_text(marker+old)
    print(table.to_string(index=False))
    print(pd.DataFrame(ar).to_string(index=False))
    print("Report:",OUT/"report.html")


if __name__=="__main__":
    main()

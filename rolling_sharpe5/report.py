"""Write the self-contained Chinese report from computed result tables."""
from pathlib import Path
from html import escape
import json

import pandas as pd

from analyze_increment import hac_mean, monthly_compound

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def table(frame):
    return frame.to_html(index=False, border=0, classes="data", escape=True)


def pct(x):
    return f"{100*x:.2f}%"


def main():
    stats = pd.read_csv(OUT / "inference.csv")
    alpha = pd.read_csv(OUT / "dynamic_pool_alpha.csv")
    counts = pd.read_csv(OUT / "yearly_pool_counts.csv")
    members = pd.read_csv(OUT / "yearly_pool_members.csv")
    primary = stats[stats.method.eq("each_year_iid") & stats.availability.eq("published_before_year") & stats.scenario.eq("5bps") & stats.hac_lags.eq(3)]
    main_rows = []
    for period in ["1990-2025", "2005-2025", "2015-2025", "2020-2025"]:
        z = primary[primary.period.eq(period)].set_index("series")
        a = alpha[alpha.method.eq("each_year_iid") & alpha.availability.eq("published_before_year") & alpha.scenario.eq("5bps") & alpha.hac_lags.eq(3) & alpha.period.eq(period)].iloc[0]
        d = z.loc["increment"]
        main_rows.append({"样本": period, "EARC扣费年化均值": pct(z.loc["paper_return", "annual_mean"]),
            "加入后年化增量": pct(d.annual_mean), "增量95%区间": f"[{pct(d.ci_low)}, {pct(d.ci_high)}]",
            "增量p": f"{d.p:.3f}", "因子池alpha（年化）": pct(a.annual_alpha), "alpha p": f"{a.p:.3f}",
            "空池月份/全部": f"{int(d.empty_pool_months)}/{int(d.n)}"})
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(OUT / "summary_chinese.csv", index=False)

    c = counts[counts.target_year.eq(2026) & counts.test.eq("two_sided") & counts.availability.eq("published_before_year")].copy()
    labels = {"each_year_iid": "每一年分别显著 / IID（主口径）", "each_year_hac1": "每一年分别显著 / HAC(1)",
              "pooled60_iid": "合并前60个月显著 / IID（不同解释）", "pooled60_hac3": "合并前60个月显著 / HAC(3)（不同解释）"}
    c["筛选方式"] = c.method.map(labels)
    c = c.rename(columns={"factor_count": "2026入选数"})[["筛选方式", "2026入选数"]]
    pool_members = members[members.method.eq("each_year_iid") & members.availability.eq("published_before_year") & members.test.eq("two_sided")]
    current_names = sorted(pool_members.loc[pool_members.target_year.eq(2026), "factor"].tolist())
    (OUT / "pool_2026_primary.json").write_text(json.dumps({
        "target_year": 2026, "training_years": [2021, 2022, 2023, 2024, 2025],
        "criterion": "Each year positive Sharpe, two-sided IID p<0.05;12months complete;publication_year<2026",
        "factor_count": len(current_names), "factors": current_names,
        "empty_pool_action": "cash", "return_data_end": "2025-12-31",
        "paper_factor_addition": "no significant positive after5bp incremental evidence in retrospective tests"
    }, indent=2), encoding="utf8")
    history = pool_members.groupby("target_year").factor.agg(", ".join).reset_index().rename(columns={"target_year": "入池年份", "factor": "入选因子"})

    source_rows = []
    for filename, label in [("paper_factor_daily.csv", "Compustat RDQ，双腿非空（主结果）"),
                            ("paper_factor_daily_ibes.csv", "I/B/E/S公告日，双腿非空"),
                            ("paper_factor_daily_min20.csv", "I/B/E/S公告日，每腿至少20只")]:
        p = HERE / filename
        if not p.exists():
            continue
        d = pd.read_csv(p, parse_dates=["date"])
        d = d[d.scenario.eq("5bps")].set_index("date")
        m = monthly_compound(d[["returns"]])["returns"]
        for period, start in [("1990-2025", "1990"), ("2005-2025", "2005"), ("2015-2025", "2015")]:
            s = hac_mean(m.loc[start:], lag=3)
            source_rows.append({"公告来源及规则": label, "期间": period,
                "年化均值": pct(s["annual_mean"]), "Sharpe": f'{s["sharpe"]:.3f}', "p": f'{s["p"]:.3f}'})
    source_table = pd.DataFrame(source_rows)
    source_table.to_csv(OUT / "event_source_sensitivity.csv", index=False)

    alt = pd.read_csv(OUT / "pooled_interpretation_sensitivity.csv")
    alt = alt[alt.availability.eq("published_before_year") & alt.scenario.eq("5bps") & alt.series.eq("increment") & alt.hac_lags.eq(3)]
    alt_table = pd.DataFrame({"筛选": alt.method.map(labels), "连续完整样本": "1990-2022",
        "年化增量": alt.annual_mean.map(pct), "p": alt.p.map(lambda v: f"{v:.3f}")})

    audit = json.loads((HERE / "paper_factor_audit.json").read_text())
    annual = pd.read_csv(OUT / "paper_annual.csv")
    cost = annual[annual.scenario.eq("5bps") & annual.year.between(2005, 2025)]
    turnover = cost.turnover.mean()
    fees = cost.cost_sum.mean()
    code_sources = """<ul>
<li><a href="../../13699_paper_YrbKK6Zn.pdf">用户提供论文</a>：核心EARC定义及Table 2；本文不沿用论文毛alpha作为扣费结果。</li>
<li>Chen, Andrew Y., and Tom Zimmermann (2022). Open Source Cross-Sectional Asset Pricing. Critical Finance Review 11(2):207-264. 使用本地Data-Release-2026.10快照。</li>
<li><a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html">statsmodels官方HAC文档</a>：Bartlett权重、有限样本修正、Student-t推断。</li>
<li>软件流程参考：Kassis, Timothy; Agarwal, Vinayak; He, Yuhuan; Patel, Darshil; Brueckner, Aubrey M. (2026). <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>。仅作为统计技能流程来源，不是投资结论的证据。</li>
</ul>"""
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>五年逐年Sharpe筛选与EARC增量检验</title>
<style>body{{font-family:system-ui,'Noto Sans CJK SC',sans-serif;max-width:1250px;margin:38px auto;padding:0 24px;color:#172536;line-height:1.75;background:#fbfcfe}}h1{{font-size:28px}}h2{{margin-top:34px;font-size:21px}}p{{max-width:1100px}}.lead{{background:#eaf0f8;padding:18px 22px;border-left:5px solid #285f99}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:14px;width:100%;background:white}}td,th{{padding:10px 12px;text-align:left;border-bottom:1px solid #dce3ec}}th{{background:#edf2f8}}img{{width:100%;max-width:1100px}}code,pre{{background:#edf2f8}}pre{{padding:16px;overflow:auto}}a{{color:#185a96}}small{{color:#536173}}</style>
<h1>五年逐年Sharpe筛选：加入EARC后是否有净增量收益</h1>
<p><small>运行日期：2026-11-03；收益数据截至2025-12-31。逐年冻结因子池；原始数据与既有结果不改动。</small></p>
<div class="lead"><b>没有证据支持扣除5bp后显著的正增量收益。</b>主口径是前五个完整年度中每一年分别正向显著；2026年入选0个因子，2009年起主口径均为空池。空池时比较退化为EARC净收益对现金，不能解释成击败一个丰富的因子库。5bp只准确作用于新建EARC持仓，原库因子仍为毛收益。</div>
<h2>结果</h2><p>1985–1989年用于最初的60个月β训练，正式比较从1990年开始。以下年化均值=12×月收益均值，月收益由每日净收益复利汇总；不是CAGR。置信区间与p值使用月度HAC(3)，双侧检验。每个因子初始总敞口为200%；组合按因子等权。</p>
<div class="scroll">{table(main_table)}</div>
<p>“加入后年化增量”比较加入前后的等权组合。因子池alpha来自同一回归中的公共截距和逐年各入选因子的独立β；β用于样本内评估显著性，不能当作样本外可交易权重。另保存仅用过去60个月拟合β的样本外对冲剩余收益。HAC(6)/(12)、不限制发表年份、额外1%年借券费及FF5+动量控制均有单独结果表。</p>
<img src="overview.png" alt="逐年因子数、EARC累计净值和增量收益置信区间">
<h2>筛选如何执行</h2>
<p>目标年份y只使用y−5至y−1的五个日历年；每年必须12个月完整，平均多空收益为正且双侧t检验p&lt;0.05。零Sharpe检验与零均值检验等价；IID正态假设下每年12个月意味着年化Sharpe须超过2.201，并连续五年满足。另做逐年HAC(1)及单侧5%筛选敏感性。缺失不补零，原论文已确定的多空方向不重新翻转。主结果还要求因子发表年份&lt;y；不加此门槛的回顾性结果也保存。这里p&lt;0.05是单因子阈值，不是212个因子的整体错误率保证。</p>
{table(c)}<p>“五年每年显著”与“合并五年整体显著”不是同一规则，不能为了获得非空池而互换。主口径1990–2026仅以下7个年份有非空池，其余年份均保留为现金：</p>{table(history)}
<h2>论文因子与5bp成本</h2>
<p>使用Compustat季度公告日RDQ及历史CCM链接，公告后第6–20个交易日做多，第36–50日做空；每日等权，每腿1元/每1元成本后NAV。公告时冻结过去90个日历日内至少5位年度EPS分析师覆盖。要求当前加前三次已知公告位于400日内，最近公告间隔70–110日。只有已知公告进入信号，不用真实未来公告。两腿非空即交易，主结果没有额外价格、市值、流动性或20只门槛。</p>
<p>每一元买入或卖出都收取0.0005元，成本=0.0005×绝对成交金额/NAV；不是每天固定扣5bp，也不是单次往返总共5bp。gross、5bp、5bp+1%年借券费分别运行独立持仓和NAV路径，解出成本后等权仓位。2005–2025年平均每年绝对成交额约为NAV的{turnover:.1f}倍，交易费用约{pct(fees)}；高换手解释了毛收益与净收益的巨大差距。</p>
<p>EARC已准确扣除此交易费。Zoo库没有股票持仓或换手，因此基准及回归控制仍为毛收益，因子配置切换/β对冲的额外费用也未计入；这是“EARC扣费后相对毛因子池”的增量检验，不是整个因子池完全扣费后的可执行组合回测。</p>
<h2>来源与实现敏感性</h2>{table(source_table)}
<p>以上各口径预先由论文规则及数据来源决定，没有按结果挑选。主结果使用RDQ；I/B/E/S结果保留作来源敏感性。论文“三个月”近似为90日；400日四次公告按包含当前解释。历史数据修订、CCM/CUSIP链接及分析师活动状态无法完全还原实时版本，故是回顾性重建。缺失持仓收益按0记账且记录，主路径共{audit['missing_held_returns']['5bps']}条；缺席证券按最后标记退出共{audit['absent_exits']['5bps']}条。融资、抵押返息、冲击、实际借券可得性及终点清仓费用未建模。</p>
<h2>合并60个月的另一种解释</h2>
<p>仅供解释差异参考，不替代主规则。2023年2–12月，年初已入选的部分期权因子没有后续收益；完整1990–2025组合因此无法无假设地计算。没有事后删掉这些因子、重新归一或补零。以下只列缺失发生前连续完整的1990–2022样本：</p>{table(alt_table)}
<h2>复现与核验</h2>
<p>验证包括：当年收益任意改动不影响当年名单或训练β；缺失历史月排除因子；已知β模拟恢复；空池与多因子组合恒等式；日收益=费用前收益−交易费−借券费；成本=换手×费率；独立二分法校验成本后NAV；事件/分析师状态无未来引用。因子池当前年份可计算到2026，但本地没有2026收益，不能检验2026实绩。</p>
<pre>cd /home/jkp/data/earnings-timing
.venv/bin/python rolling_sharpe5/audit/extend_selection_audit.py
.venv/bin/python rolling_sharpe5/build_paper_factor.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python rolling_sharpe5/analyze_increment.py
.venv/bin/python rolling_sharpe5/audit/validate_increment_analysis.py
.venv/bin/python rolling_sharpe5/report.py</pre>
<p><a href="pool_2026_primary.json">2026主因子池</a> · <a href="yearly_pool_members.csv">逐年名单</a> · <a href="yearly_pool_counts.csv">逐年数量</a> · <a href="inference.csv">完整增量统计</a> · <a href="dynamic_pool_alpha.csv">因子池alpha</a> · <a href="ff6_alpha_daily.csv">FF6 alpha</a> · <a href="monthly_comparison.csv">逐月收益</a> · <a href="design_and_validation.json">设计与指纹</a> · <a href="../paper_factor_audit.json">持仓重建审计</a></p>
<h2>来源</h2>{code_sources}</html>"""
    (OUT / "report.html").write_text(html, encoding="utf8")
    note = """ROLLING FIVE-YEAR SIGNIFICANT SHARPE / PAPER EARC ADDITION

Open results/report.html for the Chinese report and results/summary_chinese.csv
for the compact result table. Primary: every previous calendar year separately
significant (positive mean, two-sided IID p<0.05), complete12months per year,
publication year before pool year. All original data/results remain unchanged.

The2026 primary pool is empty; no2026 return data exist locally.
EARC is independently reconstructed from Compustat RDQ, historical CCM links,
CRSP daily returns, and retained90-day analyst coverage states.
Primary costs:5bp per dollar bought/sold, separate holdings/NAV, no borrow.
Zoo control returns remain GROSS because positions/turnover are unavailable.
This is NOT an all-components-net implementable factor-fund backtest.

Reproduce from /home/jkp/data/earnings-timing:
  .venv/bin/python rolling_sharpe5/audit/extend_selection_audit.py
  .venv/bin/python rolling_sharpe5/build_paper_factor.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python rolling_sharpe5/analyze_increment.py
  .venv/bin/python rolling_sharpe5/audit/validate_increment_analysis.py
  .venv/bin/python rolling_sharpe5/report.py

Sensitivity reconstruction: build_paper_factor.py --event-source ibes.
Archived initial IBES/min20 results have explicit _min20 filenames.
Main uses both legs nonempty, as paper has no minimum20 constraint.
Pooled60 tests are alternative interpretation, with2023 missing option returns;
only continuous1990-2022 comparison is reported rather than filling missing data.

Software-method sources, including the used statistical skill reference, are
listed in report.html. See audit/ for independent checks and paper definitions.
"""
    (HERE / "README.txt").write_text(note, encoding="utf8")
    print(main_table.to_string(index=False))
    print("Report:", OUT / "report.html")


if __name__ == "__main__":
    main()

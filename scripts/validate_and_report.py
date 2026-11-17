"""Validate saved artifacts, record provenance, and write the numeric report."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocp import metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, sem=None, percent=False):
    if np.isinf(value):
        return "∞"
    text = f"{value*100:.4f}%" if percent else f"{value:.6g}"
    if sem is not None and np.isfinite(sem) and sem > 0:
        text += f" ± {sem*100:.4f} pp" if percent else f" ± {sem:.3g}"
    return text


def main():
    config = json.loads((ROOT / "config.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    status = json.loads((ROOT / "STATUS.json").read_text())
    nominal = pd.read_csv(ROOT / "results/nominal_per_trial.csv")
    summary = pd.read_csv(ROOT / "results/summary.csv")
    sweep = pd.read_csv(ROOT / "results/sweep_per_trial.csv")
    comparison = pd.read_csv(ROOT / "results/paper_comparison.csv")
    assert len(nominal) == 315 and len(sweep) == 10500 and len(summary) == 72
    assert nominal.coverage.between(0,1).all() and sweep.coverage.between(0,1).all()
    assert not nominal[["coverage","mean_width","median_width","q95_width"]].isna().any().any()
    assert (nominal.mean_width >= 0).all()
    for item in manifest["files"]:
        assert digest(ROOT / item["path"]) == item["sha256"], item["path"]
        if "/raw/" in item["path"] or "/processed/" in item["path"]:
            author = ROOT / "upstream/tests/datasets" / Path(item["path"]).relative_to("data")
            assert digest(author) == item["sha256"], str(author)
    trace_checks = 0
    for dataset in config["datasets"]:
        z = np.load(ROOT / "results/traces" / f"{dataset}.npz")
        b, o = int(z["burnin"]), int(z["eval_offset"])
        rows = nominal[nominal.dataset == dataset]
        expected_n = len(z["scores"][0]) - b - o
        assert np.all(rows.n == expected_n)
        for method in rows.method.unique():
            assert not np.isnan(z[method]).any()
            for i,seed in enumerate(z["seeds"]):
                m = metrics(z["scores"][i,b+o:], z[method][i,o:])
                row = rows[(rows.method==method) & (rows.seed==seed)].iloc[0]
                for key,value in m.items():
                    assert np.isclose(value,row[key],rtol=1e-12,atol=1e-12), (dataset,method,seed,key)
                trace_checks += 1
    stock_mask = comparison.dataset.isin(["AXP","AAPL","AMZN","GOOGL"]) & comparison.method.isin(
        ["UP","UP-corrected","KT","DtACI","SFOGD","P"])
    stock = comparison[stock_mask]
    assert len(stock) == 147 and stock.match_at_published_precision.all()
    assert subprocess.check_output(["git","-C",str(ROOT/"upstream"),"status","--porcelain"], text=True) == ""
    test_log = (ROOT / "logs/pytest.log").read_text()
    assert re.search(r"\b15 passed\b",test_log), test_log
    for name in ["main_results","axp_dynamics","all_datasets_coverage","up_variants"]:
        for extension in ["png","svg"]:
            assert (ROOT / f"results/figures/{name}.{extension}").stat().st_size > 10000
    validation = dict(checked_at=datetime.now(timezone.utc).isoformat(), status="passed",
                      unit_tests_passed=15, traced_runs_recomputed=trace_checks,
                      nominal_runs=315, sweep_runs=10500, grid_runs=595,
                      stock_published_cells_matched=147, stock_published_cells_checked=147,
                      all_data_checksums_valid=True, author_raw_and_forecast_files_match=True,
                      upstream_unmodified=True, figures_present=8,
                      limitations="Validation confirms this implementation and its documented scope, not exact reproduction of every paper result.")
    (ROOT / "VALIDATION.json").write_text(json.dumps(validation,indent=2))
    status["exact_stock_cells_including_p"] = 147
    status["validation"] = "passed; 15 tests, 315 trace/metric checks, 147 published stock cells"
    (ROOT / "STATUS.json").write_text(json.dumps(status,indent=2))

    commit = status["upstream_commit"]
    sources = [
        ("paper/source.pdf", "https://raw.githubusercontent.com/mlresearch/v306/main/assets/liu26y/liu26y.pdf", "Published PMLR paper"),
        ("reference/pid/methods.py", "https://raw.githubusercontent.com/aangelopoulos/conformal-time-series/b729c3f5ff633bfc43f0f7ca08199b549c2573ac/core/methods.py", "Unedited baseline reference"),
        ("reference/pid/LICENSE", "https://raw.githubusercontent.com/aangelopoulos/conformal-time-series/b729c3f5ff633bfc43f0f7ca08199b549c2573ac/LICENSE", "MIT notice"),
        ("reference/elec2.pkl", f"https://raw.githubusercontent.com/AdaMLLab/up-for-ocp/{commit}/tests/results/elec2.pkl", "Diagnostic only; not consumed by reproduction runner"),
        ("reference/elec2_ar_legacy.npz", "https://raw.githubusercontent.com/AdaMLLab/up-for-ocp/3148166d91fd0df068e70b5416028c67216caae7/tests/datasets/processed/elec2/ar-1-steps.npz", "Historical cache diagnostic only"),
        ("reference/elec2_prophet_legacy.npz", "https://raw.githubusercontent.com/AdaMLLab/up-for-ocp/3148166d91fd0df068e70b5416028c67216caae7/tests/datasets/processed/elec2/prophet.npz", "Historical cache diagnostic only; not substituted for AR"),
        ("reference/author_experiments.ipynb", "https://raw.githubusercontent.com/AdaMLLab/up-for-ocp/adc72fa7ff532e2141264ac01694e17f6479469d/electricity_demand.ipynb", "Historical notebook inspected; not executed"),
    ]
    for relative in ["core/methods.py","core/functional.py","core/runner.py","core/model_scores.py",
                     "tests/datasets.py","tests/inset_plot.py","tests/make_plots.sh","LICENSE"]:
        sources.append(("upstream/"+relative,f"https://raw.githubusercontent.com/AdaMLLab/up-for-ocp/{commit}/{relative}","Unedited official source"))
    provenance = dict(paper_page="https://proceedings.mlr.press/v306/liu26y.html",
                      author_commit=commit, dataset_manifest="data/manifest.json",
                      dataset_source_base=manifest["raw_source"],
                      sources=[dict(path=p,url=u,role=r,sha256=digest(ROOT/p)) for p,u,r in sources if (ROOT/p).exists()],
                      reproduction_files={str(p.relative_to(ROOT)):digest(p) for p in
                                          [ROOT/"ocp.py",ROOT/"config.json",ROOT/"requirements.lock.txt",*sorted((ROOT/"scripts").glob("*.py")),ROOT/"tests/test_ocp.py"]})
    (ROOT / "PROVENANCE.json").write_text(json.dumps(provenance,indent=2))

    lines = ["# 复现实验完整结果", "", "由 `scripts/validate_and_report.py` 从本次 CSV 输出生成。", "",
             "四只股票五种公开基线及 AXP 校正版：147/147 个数值在论文报告精度下匹配。数值按三位有效数字核对；原始精度另存 CSV。", "",
             "合成数据为 10 次独立运行的指标均值，± 表示一个标准误；股票和电力只有一条序列。∞ 保留，不删除。PI-reconstructed 是配置不完整时的替代实现，不能称为原论文 PI 的精确复现。", "",
             "本次使用作者的基础模型预测缓存，重新计算所有 conformal 半径，没有读取作者预计算的 conformal 输出作为实验结果。未重训基础预测模型。", "",
             "## 论文值匹配情况", "", "随机合成实验的逐位匹配不是统计复现的必要条件；下面仅列诊断计数。电力和 PI 的不匹配详见 README。", "",
             "| 数据集 | 方法 | 匹配数 / 论文数值数 |", "|---|---|---:|"]
    for (dataset,method), part in comparison.groupby(["dataset","method"],sort=False):
        lines.append(f"| {dataset} | {method} | {part.match_at_published_precision.sum()} / {len(part)} |")
    for dataset in config["datasets"]:
        lines += ["",f"## {dataset}","",
                  "| 方法 | 覆盖率 | 平均宽度 | 中位宽度 | 95% 分位宽度 | 无限区间比例 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for _,r in summary[summary.dataset==dataset].iterrows():
            lines.append("| "+" | ".join([r.method,fmt(r.coverage,r.coverage_sem,True),
                         fmt(r.mean_width,r.mean_width_sem),fmt(r.median_width,r.median_width_sem),
                         fmt(r.q95_width,r.q95_width_sem),fmt(r.infinite_fraction,percent=True)])+" |")
    lines += ["", "## 图", "", "![全部数据集覆盖率](results/figures/all_datasets_coverage.png)", "",
              "![AXP 动态](results/figures/axp_dynamics.png)", "",
              "完整协议、四个 UP 版本的区别、无法对齐部分及重跑命令见 [README](README.md)。", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(validation,indent=2))


if __name__ == "__main__":
    main()

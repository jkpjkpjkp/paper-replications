"""Fresh full-universe ten-seed frozen evaluation with explicit adaptations.

Official source is imported read-only. Data are current provider vintage.
The release constant-score runner defaults to approximate IC whereas the main
checkpoint configs set exact IC. Both constant IC settings are reported; the
release-native approximate setting is the declared primary artifact comparison.
"""
from pathlib import Path
from runtime_info import cgroup_directory, resource_limits, peak_memory
import dataclasses
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
UP = ROOT / "upstream"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "fresh_frozen"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(UP / "src"))
import CON2 as con
import NodeMechanism as nm

SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
con.torch.set_num_threads(2)


def source_config(seed):
    path = UP / "checkpoints/source_50path" / f"seed_{seed}" / "run_summary.json"
    # Exact upstream summaries have UTF-8 BOM; upstream json.load(utf-8) fails.
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    return con.ExperimentConfig(**obj["config"])


def finalize_data():
    vix = RAW / "VIXCLS.parquet"
    if not vix.exists():
        source = ROOT / "data/provider_bytes/CBOE_VIX_History.csv"
        frame = pd.read_csv(source)
        out = pd.DataFrame({"date": pd.to_datetime(frame.DATE), "VIXCLS": frame.CLOSE})
        out = out[out.date <= "2026-03-31"]
        out.to_parquet(vix, index=False)
        manifest = json.loads((ROOT / "data/manifest.json").read_text())
        manifest["VIX_adaptation"] = {"url": "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "reason": "FRED VIXCLS and VXVCLS timed out; use primary Cboe daily VIX closing values as the same index input. Historical provider-vintage equality to author's FRED parquet is unverified.", "rows": len(out), "VXVCLS": "omitted; load-only variable, not used in model/state features"}
        manifest["ready"] = True
        (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size, "symlink": p.is_symlink()} for p in sorted(RAW.iterdir()) if p.is_file()]
    (OUT / "data_hashes.json").write_text(json.dumps(files, indent=2) + "\n")
    return con.load_all_data(RAW)


def nodes(es, cfg, source_tables, dataset, seed, details):
    data = con.torchify_episode_set(es)
    weights = {}
    for label, penalty in [("Feature", "none"), ("Graph", "tail_cvar")]:
        name = ("ETF_" if dataset == "etf" else "") + "GraphSignal-ICL" + ("-NoPenalty" if label == "Feature" else "")
        net = con.GraphContractNet(9, cfg.hidden_dim, cfg, use_graph=False)
        net.load_state_dict(con.torch.load(source_tables / (name + "_model.pt"), map_location="cpu", weights_only=True))
        net.eval()
        with con.torch.no_grad():
            th, et, la = net(con.select_torch_features(data, "graph_scores"), con.choose_torch_W(data, "equal"))
        weights[label] = (th.numpy(), et.numpy(), la.numpy() if penalty != "none" else np.zeros_like(la.numpy()))
    rows = []
    for b, date in enumerate(es.dates):
        sf, sg = None, None
        for label in ["Feature", "Graph"]:
            th, et, la = [a[b] for a in weights[label]]
            _, ss = con.solve_best_response_np(th, et, la, es.W_gls[b], es.m_tail_cvar[b], es, b, cfg)
            if label == "Feature":
                sf = ss
            else:
                sg = ss
        # The released mechanism script replaces node-specific coefficients by
        # each state's cross-node means. Save that separate construction too.
        means = {}
        for label in ["Feature", "Graph"]:
            th, et, la = [np.full(es.N, a[b].mean()) for a in weights[label]]
            _, means[label] = con.solve_best_response_np(th, et, la, es.W_gls[b], es.m_tail_cvar[b], es, b, cfg)
        for j, symbol in enumerate(es.symbols):
            rows.append({"dataset": dataset, "seed": seed, "date": date.date().isoformat(), "node": j, "symbol": symbol, "m_tail_cvar": es.m_tail_cvar[b, j], "s_feature_only": sf[j], "s_graph_penalty": sg[j], "delta_graph_minus_feature": sg[j] - sf[j], "released_mean_coefficient_delta": means["Graph"][j] - means["Feature"][j]})
    return pd.DataFrame(rows)


def temporal_audit(data, seed):
    cfg = source_config(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for ds, prep, start, train_end, val_start, val_end, test_start, end in [
        ("industry", con.prepare_industry_returns, cfg.industry_start, cfg.train_end, cfg.val_start, cfg.val_end, cfg.test_start, cfg.test_end),
        ("etf", con.prepare_etf_returns, cfg.etf_start, cfg.etf_train_end, cfg.etf_val_start, cfg.etf_val_end, cfg.etf_test_start, cfg.etf_test_end),
    ]:
        rets, factor, vix = prep(data, cfg)
        es = con.build_episode_set(ds, rets, factor, vix, cfg, start, end, rng)
        assert es.N == (49 if ds == "industry" else 20)
        expected_counts = [190, 25, 228] if ds == "industry" else [28, 13, 77]
        for split, s, e, expected in [("train", start, train_end, expected_counts[0]), ("validation", val_start, val_end, expected_counts[1]), ("test", test_start, end, expected_counts[2])]:
            subset = es.subset_by_date(s, e)
            assert subset.B == expected, (ds, split, subset.B, expected)
            crossed = []
            for d in subset.dates:
                ix = rets.index.get_loc(d)
                last = rets.index[ix + cfg.episode_horizon - 1]
                if last > pd.Timestamp(e):
                    crossed.append({"decision_date": str(d.date()), "label_end": str(last.date())})
            rows.append({"dataset": ds, "split": split, "N": es.N, "states": subset.B, "first": str(subset.dates[0].date()), "last": str(subset.dates[-1].date()), "forward_labels_past_split_end": crossed, "VIX_timing": "source uses VIX at decision date, while return lookback stops previous day; requires end-of-day-state interpretation, not usable as pre-open trading signal"})
    (OUT / "temporal_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    started = time.monotonic()
    log = (ROOT / "logs/fresh-frozen.log").open("a", buffering=1)
    console = sys.stdout
    class Tee:
        def write(self, text):
            console.write(text)
            log.write(text)
        def flush(self):
            console.flush()
            log.flush()
    sys.stdout = Tee()
    group = cgroup_directory()
    (OUT / "kernel_limits.json").write_text(json.dumps(resource_limits(group), indent=2))
    data = finalize_data()
    temporal_audit(data, 7)
    all_nodes = []
    constant_sensitivity = []
    for seed in SEEDS:
        target = OUT / f"seed_{seed}"
        tables = target / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        marker = target / "complete.json"
        if marker.exists():
            all_nodes.append(pd.read_csv(target / "node_exposure.csv"))
            constant_sensitivity.extend(json.loads((target / "constant_ic_sensitivity.json").read_text()))
            continue
        print(f"Building full 50/500-path industry+ETF states for seed {seed}", flush=True)
        cfg = source_config(seed)
        con.set_seed(seed)
        cfg_eval = dataclasses.replace(cfg, mc_paths=500, torch_threads=2)
        industry, etf = con.build_eval_tests_with_frozen_random_controls(data, cfg, cfg_eval)
        node_parts = []
        sensitivity = []
        for ds, es in [("industry", industry), ("etf", etf)]:
            assert es.N == (49 if ds == "industry" else 20)
            assert es.B == (228 if ds == "industry" else 77)
            rows, details = [], []
            modeldir = UP / "checkpoints/source_50path" / f"seed_{seed}" / "tables"
            for spec in con.neural_contract_specs(cfg_eval, ds):
                label = spec["label"]
                const = "ConstantPenalty" in label
                source_dir = UP / "results/constant_score_500" / f"seed_{seed}" / "source_50path_tables" if const else modeldir
                # Faithful to the source constant-score runner's default; preserve
                # an exact-system sensitivity separately, never select by result.
                variant_cfg = dataclasses.replace(cfg_eval, neural_exact_ic=False) if const else cfg_eval
                print(f"Evaluating {ds} seed={seed}: {label}, exact_IC={variant_cfg.neural_exact_ic}", flush=True)
                metric, detail = con.evaluate_saved_neural_contract(es, variant_cfg, source_dir, **spec)
                metric.update(dataset=ds, neural_exact_ic=variant_cfg.neural_exact_ic)
                rows.append(metric)
                details.append(detail)
                if const:
                    corrected, _ = con.evaluate_saved_neural_contract(es, cfg_eval, source_dir, **spec)
                    corrected.update(dataset=ds, seed=seed, neural_exact_ic=True, source_training_exact_ic="unrecorded; published runner default=False")
                    sensitivity.append(corrected)
                if seed == 7 and ds == "industry" and label == "GraphSignal-ICL":
                    repeated, _ = con.evaluate_saved_neural_contract(es, variant_cfg, source_dir, **spec)
                    keys = ["CVaR95_loss", "crowding", "mean_payoff", "co_crash_freq"]
                    assert all(repeated[k] == metric[k] for k in keys)
                    (OUT / "deterministic_rerun.json").write_text(json.dumps({k: {"first": metric[k], "second": repeated[k], "identical": True} for k in keys}, indent=2))
            frame = pd.DataFrame(rows)
            con.write_display_csv(frame, tables / f"{ds}_neural_results.csv")
            con.write_display_csv(frame, tables / f"{ds}_main_results_all.csv")
            pd.concat(details).to_csv(tables / f"{ds}_episode_details_neural.csv", index=False)
            node_parts.append(nodes(es, cfg_eval, modeldir, ds, seed, details))
        node_df = pd.concat(node_parts, ignore_index=True)
        node_df.to_csv(target / "node_exposure.csv", index=False)
        all_nodes.append(node_df)
        constant_sensitivity.extend(sensitivity)
        (target / "constant_ic_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
        marker.write_text(json.dumps({"seed": seed, "status": "complete", "scope": "fresh full-universe frozen checkpoint evaluation; current provider vintage", "elapsed_seconds": time.monotonic() - started}, indent=2))
    con.aggregate_multi_seed_outputs(OUT, SEEDS)
    pd.DataFrame(constant_sensitivity).to_csv(OUT / "constant_exact_IC_sensitivity.csv", index=False)
    node_df = pd.concat(all_nodes, ignore_index=True)
    node_df = nm.assign_bins(node_df)
    node_df.to_csv(OUT / "node_exposure_all.csv", index=False)
    nm.summarize_bins(node_df, "delta_graph_minus_feature").to_csv(OUT / "actual_node_coefficient_mechanism_bins.csv", index=False)
    nm.summarize_bins(node_df, "released_mean_coefficient_delta").to_csv(OUT / "released_mean_coefficient_mechanism_bins.csv", index=False)
    manifest = {"seeds": SEEDS, "datasets": {"industry": {"N": 49, "test_states": 228}, "etf": {"N": 20, "test_states": 77}}, "M_paths": 500, "source_training_M_paths": 50, "checkpoint_evaluations": 140, "additional_declared_constant_exact_IC_evaluations": 20, "deterministic_check": "seed7 industry Graph,4 metrics bitwise identical", "node_rows": len(node_df), "elapsed_seconds": time.monotonic() - started, "adaptations": ["current French/Yahoo vintage, original sample cutoffs", "Cboe daily VIX close replaces inaccessible FRED VIX parquet", "UTF-8 BOM normalized only while reading source config; upstream code preserved", "constant-score default approximate-IC and exact-IC sensitivity both reported", "node-wise coefficient mechanism and released state-mean coefficient construction both reported"], "limitations": ["source randomized-score training provenance unresolved", "source constant-score training exact-IC setting unrecorded; published runner defaults False", "frozen checkpoint replay does not retrain original models", "source decision-date split has unpurged forward labels; inspect temporal_audit.json", "source VIX decision-date availability is not pre-open chronological trading evidence"]}
    (OUT / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "memory_peak_bytes.txt").write_text(str(peak_memory(group)) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

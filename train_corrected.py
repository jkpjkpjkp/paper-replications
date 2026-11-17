"""Complete ten-seed training/evaluation, with declared chronology corrections.

All original learned comparisons and both released auxiliary graph baselines
are retained. Source code and pretrained artifacts remain unchanged. This is a
corrected methodological reproduction, not a literal paper-table replication.
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
OUT = ROOT / "results" / "corrected_training"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(UP / "src"))
import CON2 as con
import NodeMechanism as nm
from run_frozen import source_config, nodes

SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
con.torch.set_num_threads(1)


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def splits(es, returns, cfg, dataset):
    if dataset == "industry":
        start, tr_end, va_start, va_end, te_start, end = cfg.industry_start, cfg.train_end, cfg.val_start, cfg.val_end, cfg.test_start, cfg.test_end
    else:
        start, tr_end, va_start, va_end, te_start, end = cfg.etf_start, cfg.etf_train_end, cfg.etf_val_start, cfg.etf_val_end, cfg.etf_test_start, cfg.etf_test_end
    def cutoff(last):
        ix = returns.index.searchsorted(pd.Timestamp(last), side="right") - 1
        return str(returns.index[ix - cfg.episode_horizon + 1].date())
    train = es.subset_by_date(start, cutoff(tr_end), dataset + "_train_purged")
    val = es.subset_by_date(va_start, cutoff(va_end), dataset + "_val_purged")
    test = es.subset_by_date(te_start, end, dataset + "_test")
    train, val, test, scaler = con.standardize_feature_sets(train, val, test)
    for subset, boundary in [(train, tr_end), (val, va_end)]:
        for date in subset.dates:
            ix = returns.index.get_loc(date)
            assert returns.index[ix + cfg.episode_horizon - 1] <= pd.Timestamp(boundary)
    return train, val, test, scaler


def build(data, cfg):
    source_rng = np.random.default_rng(cfg.random_seed)
    eval_rng = np.random.default_rng(cfg.random_seed)
    out = {}
    cfg_eval = dataclasses.replace(cfg, mc_paths=500)
    for ds, prep, start, end in [("industry", con.prepare_industry_returns, cfg.industry_start, cfg.test_end), ("etf", con.prepare_etf_returns, cfg.etf_start, cfg.etf_test_end)]:
        returns, factor, vix = prep(data, cfg)
        source = con.build_episode_set(ds, returns, factor, vix, cfg, start, end, source_rng)
        evaluation = con.build_episode_set(ds, returns, factor, vix, cfg_eval, start, end, eval_rng)
        train, val, test, scaler = splits(source, returns, cfg, ds)
        _, _, eval_test, _ = splits(evaluation, returns, cfg_eval, ds)
        eval_test = con.copy_random_controls(test, eval_test)
        assert (train.B, val.B, test.B, test.N) == ((189, 24, 228, 49) if ds == "industry" else (28, 12, 77, 20))
        out[ds] = (train, val, test, eval_test)
    return out


def fit(seed, ds, cfg, spec, sets, table_dir):
    label = spec["label"]
    marker = table_dir / (label + "_training_complete.json")
    rng_path = table_dir / (label + "_rng_after.pt")
    if marker.exists():
        con.torch.set_rng_state(con.torch.load(rng_path, weights_only=True))
        return
    save(OUT / "progress.json", {"status": "training", "seed": seed, "dataset": ds, "method": label, "epochs": 250, "mc_paths": 50})
    print(f"TRAIN seed={seed} {ds} {label}: full250epoch limit,50paths,exactIC, purged labels", flush=True)
    started = time.monotonic()
    train, val, test, _ = sets
    metric, history, details = con.train_neural_contract(train, val, test, cfg, table_dir, **spec)
    history.to_csv(table_dir / (label + "_training_log.csv"), index=False)
    checkpoint = table_dir / (label + "_model.pt")
    con.torch.save(con.torch.get_rng_state(), rng_path)
    save(marker, {"seed": seed, "dataset": ds, "method": label, "actual_epochs": int(history.epoch.max()), "elapsed_seconds": time.monotonic() - started, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "source_code_sha256": hashlib.sha256((UP / "src/CON2.py").read_bytes()).hexdigest(), "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config": dataclasses.asdict(cfg), "splits": {"train": train.B, "validation": val.B, "test": test.B}, "point_in_time": "VIX prior available trading close; labels purged at train/validation end; feature normalization fit only on purged training states", "validation_selection": "unchanged source policy objective; epochs1,every5,and final; no test selection"})
    print(f"DONE {label}, {time.monotonic() - started:.1f}s", flush=True)


def main():
    console = sys.stdout
    log = (ROOT / "logs/corrected-training.log").open("a", buffering=1)
    class Tee:
        def write(self, value):
            console.write(value)
            log.write(value)
        def flush(self):
            console.flush()
            log.flush()
    sys.stdout = Tee()
    started = time.monotonic()
    group = cgroup_directory()
    save(OUT / "kernel_limits.json", resource_limits(group))
    data = con.load_all_data(ROOT / "data/raw")
    data["vix"] = data["vix"].copy()
    data["vix"]["VIXCLS"] = data["vix"]["VIXCLS"].shift(1)
    node_parts = []
    for seed in SEEDS:
        target = OUT / f"seed_{seed}"
        tables = target / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        if (target / "complete.json").exists():
            node_parts.append(pd.read_csv(target / "node_exposure.csv"))
            continue
        cfg = dataclasses.replace(source_config(seed), neural_exact_ic=True, torch_threads=1)
        sets = build(data, cfg)
        con.set_seed(seed)
        # Match the original main-run initializer order; constant was a separate
        # runner in the release and is trained below with its original seed reset.
        for ds in ["industry", "etf"]:
            for spec in con.neural_contract_specs(cfg, ds):
                if "ConstantPenalty" not in spec["label"]:
                    fit(seed, ds, cfg, spec, sets[ds], tables)
        con.set_seed(seed)
        for ds in ["industry", "etf"]:
            spec = next(s for s in con.neural_contract_specs(cfg, ds) if "ConstantPenalty" in s["label"])
            fit(seed, ds, cfg, spec, sets[ds], tables)
        cfg_eval = dataclasses.replace(cfg, mc_paths=500)
        seed_nodes = []
        for ds in ["industry", "etf"]:
            es = sets[ds][3]
            rows, details = [], []
            for spec in con.neural_contract_specs(cfg_eval, ds):
                metrics, detail = con.evaluate_saved_neural_contract(es, cfg_eval, tables, **spec)
                metrics.update(dataset=ds, neural_exact_ic=True)
                rows.append(metrics)
                details.append(detail)
            frame = pd.DataFrame(rows)
            con.write_display_csv(frame, tables / f"{ds}_neural_results.csv")
            con.write_display_csv(frame, tables / f"{ds}_main_results_all.csv")
            pd.concat(details).to_csv(tables / f"{ds}_episode_details_neural.csv", index=False)
            seed_nodes.append(nodes(es, cfg_eval, tables, ds, seed, details))
        node_frame = pd.concat(seed_nodes, ignore_index=True)
        node_frame.to_csv(target / "node_exposure.csv", index=False)
        node_parts.append(node_frame)
        save(target / "run_summary.json", {"config": dataclasses.asdict(cfg), "corrections": ["purged labels", "lagged VIX", "constant exactIC", "current provider vintage"]})
        save(target / "complete.json", {"seed": seed, "models_trained": 14, "frozen_paths": 500, "full_universes": [49, 20]})
    con.aggregate_multi_seed_outputs(OUT, SEEDS)
    frame = nm.assign_bins(pd.concat(node_parts, ignore_index=True))
    frame.to_csv(OUT / "node_exposure_all.csv", index=False)
    nm.summarize_bins(frame, "delta_graph_minus_feature").to_csv(OUT / "actual_node_coefficient_mechanism_bins.csv", index=False)
    save(OUT / "summary.json", {"status": "complete_corrected_adaptation", "original_paper_replication_status": "partial", "seeds": SEEDS, "models_trained": 140, "original_required_models": "5 variants x2universes x10seeds=100", "auxiliary_release_models": "2 additional graph variants x2universes x10seeds=40", "training_paths": 50, "evaluation_paths": 500, "epoch_limit": 250, "source_early_stopping_patience": 50, "test_states": {"industry": 228, "ETF": 77}, "purged_train_validation_states": {"industry": [189, 24], "ETF": [28, 12]}, "no_test_tuning": True, "original_hyperparameters_retained": True, "corrections": ["20day labels purged across split boundaries", "VIX feature lagged one available trading close", "constant-score analytic feasibility exact during training and evaluation", "current Yahoo/French vintage and primaryCboe VIX, source manifest records differences", "node exposures recomputed using asset-specific learned coefficients"], "elapsed_seconds": time.monotonic() - started, "memory_peak_bytes": peak_memory(group)})
    save(OUT / "progress.json", {"status": "complete", "models_trained": 140})
    print("ALL140 TRAINED AND500-PATH EVALUATED", flush=True)


if __name__ == "__main__":
    main()

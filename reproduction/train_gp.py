"""Released GP search (1000 individuals x 40 generations), without test-time callbacks."""
import argparse
from collections import Counter
import difflib
import json
from pathlib import Path
import time
import types

from common import ROOT, UPSTREAM, LocalStockData, initialize, metrics, fresh_output
from integration import ForwardReturn
from evaluate import portfolio_proxy
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--population', type=int, default=1000)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--train-end', choices=['2020-11-30', '2020-12-31'], default='2020-11-30')
    parser.add_argument('--legacy-return-proxy', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = fresh_output(args.output)
    torch.set_num_threads(args.threads)
    initialize()
    source = (UPSTREAM/'train_GP.py').read_text()
    patched = source.replace("train_end_time = f'{args.train_end_year}-12-31'", f"train_end_time = '{args.train_end}'")
    patched = patched.replace('target = Ref(close, -20) / close - 1', 'target = ForwardReturn(20)')
    patched = patched.replace("save_dir = f'out_gp/{args.instruments}_{args.train_end_year}_{args.freq}_{args.seed}'", "save_dir = args.output")
    (out/'gp-integration.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True), fromfile='upstream/train_GP.py', tofile='runtime/train_GP.py')))
    (out/'runtime').mkdir(exist_ok=True)
    (out/'runtime/train_GP.py').write_text(patched)
    module = types.ModuleType('local_train_gp')
    exec(compile(patched, str(out/'runtime/train_GP.py'), 'exec'), module.__dict__)
    module.StockData = LocalStockData
    module.ForwardReturn = ForwardReturn
    module.normalize_by_day = lambda value: module.data.normalize(value)
    original_estimator = module.SymbolicRegressor
    estimators = []
    def estimator_factory(**kwargs):
        kwargs.update(population_size=args.population, generations=args.generations,
                      tournament_size=min(600, args.population))
        estimator = original_estimator(**kwargs)
        estimators.append(estimator)
        return estimator
    module.SymbolicRegressor = estimator_factory
    started = time.monotonic()
    history = []
    def generation_finished():
        module.generation += 1
        best = Counter(module.cache).most_common(1)
        entry = dict(generation=module.generation, unique_evaluated=len(module.cache),
                     best_train_ic=float(best[0][1]) if best else None,
                     best_expression=best[0][0] if best else None,
                     wall_seconds=time.monotonic()-started)
        history.append(entry)
        (out/'progress.json').write_text(json.dumps(dict(status='running', generations=args.generations, **entry), indent=2))
        (out/'search_cache.json').write_text(json.dumps(module.cache))
        print(json.dumps(entry), flush=True)
    module.ev = generation_finished
    protocol = dict(method='upstream bundled gplearn symbolic GP', seed=args.seed,
                    population=args.population, generations=args.generations,
                    tournament_size=min(600, args.population), max_parenthesis_count=20,
                    features=['open','close','high','low','volume'],
                    training=['2010-01-01',args.train_end], validation=['2021-01-01','2021-12-31'],
                    test=['2022-01-01','2024-12-31'], horizon=20,
                    selection='Highest training IC in cache after fixed final generation; single expression as released GP notebook',
                    callbacks='Training diagnostics only; validation and test not used for evolution or selection')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    module.run(argparse.Namespace(cuda='', seed=args.seed, train_end_year=2020,
                                  instruments='csi300', freq='day', output=str(out)))
    (out/'data-provenance.json').write_text(json.dumps({key: value.provenance for key, value in
        [('training', module.data), ('validation', module.data_valid), ('test', module.data_test)]}, indent=2))
    best_key, best_score = Counter(module.cache).most_common(1)[0]
    expression = eval(best_key, module.__dict__)
    results = {}
    for label, data in [('validation',module.data_valid), ('test',module.data_test)]:
        with torch.no_grad():
            prediction = expression.evaluate(data).numpy()
            target = module.target.evaluate(data).numpy()
        results[label] = metrics(prediction, target)
        proxy, returns = (portfolio_proxy(prediction, target) if args.legacy_return_proxy else (None, np.array([])))
        results[label+'_return_proxy'] = proxy
        dates = data._dates[data.max_backtrack_days:-data.max_future_days]
        np.savez_compressed(out/f'{label}_predictions.npz', predictions=prediction, targets=target,
                            dates=dates.to_numpy(), stocks=np.asarray(data._stock_ids,dtype=str), proxy_returns=returns)
    (out/'best_expression.json').write_text(json.dumps(dict(expression=str(expression), generic_expression=best_key, train_ic=best_score), indent=2))
    cache_top = Counter(module.cache).most_common(50)
    (out/'pool.json').write_text(json.dumps(dict(exprs=[str(eval(key,module.__dict__)) for key,score in cache_top], weights=[score for key,score in cache_top]), indent=2))
    result = dict(status='complete', generations_completed=module.generation,
                  population=args.population, unique_evaluated=len(module.cache),
                  best_expression=str(expression), best_train_ic=best_score,
                  **results, history=history, wall_seconds=time.monotonic()-started,
                  limitations=['Community data, one seed, CSI300 only',
                               'Final training-IC-selected single factor; no notebook refitting on test data',
                               f'Training endpoint {args.train_end}; nominal December endpoint allows labels into validation',
                               'No executable backtest; optional legacy return proxies are diagnostic only'])
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (out/'progress.json').write_text(json.dumps(dict(status='complete', generations_completed=module.generation), indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

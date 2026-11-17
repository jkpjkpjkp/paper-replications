"""Re-evaluate saved formulas without retraining or executing arbitrary text."""
import argparse
import ast
import json
import re
from pathlib import Path
import numpy as np
import torch
from implementation import LocalStockData, CachedCalculator, daily_metrics
from alphagen.data.expression import Feature, FeatureType, Ref, Constant
from alphagen.config import OPERATORS

def parse_expression(text):
    classes = {op.__name__: op for op in OPERATORS}
    classes['Constant'] = Constant
    features = {f.name.lower(): Feature(f) for f in FeatureType}
    tree = ast.parse(re.sub(r'\$([a-z]+)', r'feature_\1', text), mode='eval')
    def visit(node):
        if isinstance(node, ast.Name) and node.id.startswith('feature_'):
            return features[node.id[8:]]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = visit(node.operand)
            if type(value) in (int, float):
                return -value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in classes and not node.keywords:
            return classes[node.func.id](*[visit(arg) for arg in node.args])
        raise ValueError('Unsupported expression syntax')
    return visit(tree.body)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--pool', choices=['selected', 'final'], default='selected')
    args = p.parse_args()
    folder = Path(args.run)
    manifest = json.loads((folder/'manifest.json').read_text())
    cfg = manifest['config']
    calculator_class = CachedCalculator
    if cfg.get('method') == 'paper':
        from paper_implementation import configure_paper_operators, PaperCalculator
        configure_paper_operators()
        calculator_class = PaperCalculator
    saved = json.loads((folder/('selected_pool.json' if args.pool == 'selected' else 'final_pool.json')).read_text())
    exprs = [parse_expression(s) for s in saved['exprs']]
    import qlib
    qlib.init(provider_uri=cfg['provider'], region='cn', kernels=1)
    torch.set_num_threads(cfg['threads'])
    data = LocalStockData(cfg['universe'], *cfg['splits']['test'], device=torch.device(cfg['device']))
    close = Feature(FeatureType.CLOSE)
    calc = calculator_class(data, Ref(close, -20)/close-1)
    summary, daily = daily_metrics(data, calc, exprs, saved['weights'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    daily.to_csv(output/'daily_metrics.csv', index=False)
    values = calc.make_ensemble_alpha(exprs, saved['weights']).cpu().numpy()
    observed = np.ones_like(values, dtype=bool)
    for expression in exprs:
        observed &= torch.isfinite(expression.evaluate(data)).cpu().numpy()
    np.savez_compressed(output/'predictions.npz', prediction=values,
                        observed=observed,
                        dates=np.asarray(data.eval_dates.astype(str), dtype=str),
                        instruments=np.asarray(data._stock_ids.astype(str), dtype=str),
                        feature_convention='daily standardized factors; unavailable components zero-filled in prediction and explicitly excluded by observed mask')
    (output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

"""Re-evaluate saved forecasts with opening-price rather than closing-price limits.

The released demo combines open execution with Qlib's close-to-close change
filter. Preserve that reconstruction, but report this separate timing sensitivity.
"""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import reproduce_pretrained as reproduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    original=args.run
    output=original/'open-limit-sensitivity'
    output.mkdir(exist_ok=True)
    for name in ['samples.csv','qlib_adjusted','evidence']:
        dest=output/name
        if not dest.exists():
            dest.symlink_to(original/name,target_is_directory=(original/name).is_dir())
    protocol=json.loads((original/'protocol.json').read_text())
    protocol['limit_threshold']=['$open / Ref($close, 1) - 1 >= 0.095',
                                 '$open / Ref($close, 1) - 1 <= -0.095']
    protocol['limitations'].append('Sensitivity: opening-price limit proxy replaces the demo close-to-close limit filter; official author execution remains unknown.')
    reproduction.write_json(output/'protocol.json',protocol)
    module=importlib.import_module('qlib.backtest')
    released_backtest=module.backtest

    def at_open(**kwargs):
        kwargs['exchange_kwargs']['limit_threshold']=tuple(protocol['limit_threshold'])
        return released_backtest(**kwargs)

    module.backtest=at_open
    completed=[]
    for seed in protocol['seeds']:
        source=original/f'seed-{seed}'
        if not (source/'metrics.json').exists():
            continue
        completed.append(seed)
        dest=output/source.name
        dest.mkdir(exist_ok=True)
        for shard in source.glob('predictions-*.npz'):
            target=dest/shard.name
            if not target.exists():
                target.symlink_to(shard)
        if not (dest/'metrics.json').exists():
            reproduction.backtest(SimpleNamespace(output=output,seeds=protocol['seeds']),seed)
    print('Completed opening-limit sensitivity seeds:',completed)


if __name__=='__main__':
    main()

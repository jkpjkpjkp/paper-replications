#!/usr/bin/env python3
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from krrdp.diagnostics import error_decomposition, mc_scaling

with threadpool_limits(2):
    rows = [error_decomposition(n=n, m=m, seed=20261106+r)
            for n in [32, 64, 128, 256] for m in [8, 32, 128] for r in range(20)]
    output = dict(description='Exact finite-state validation, not a theorem proof or paper option experiment',
                  decomposition=rows, mc_scaling=mc_scaling())
    Path('results/diagnostics.json').write_text(json.dumps(output, indent=2)+'\n')
    violations = []
    for row in rows:
        for s in row['stages']:
            if s['total_mse'] > s['triangle_bound']+1e-13 or s['propagation_mse'] > s['contraction_bound']+1e-13:
                violations.append(s)
    print(f'{len(rows)} replications; {len(rows)*9} stage checks; {len(violations)} violations')

#!/usr/bin/env python3
"""Extract only proposed-method rows of Table 6 from pdftotext output."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT/"sources/paper.txt").read_text()
block = text.split("Table 6. MER,", 1)[1].split("Table 7. Average", 1)[0]
matches = re.findall(r"^\s*(STCO[12])\s+([\d.]+(?:\s+[\d.]+){11})\s*$", block, re.M)
assert len(matches) == 6, "Expected two strategy rows for each of three factors"
rows = {}
for i, (strategy, numbers) in enumerate(matches):
    metric = ["paper_MER", "paper_alpha", "paper_beta"][i//2]
    values = [float(s) for s in numbers.split()]
    assert len(values) == 12
    for j, value in enumerate(values):
        key = (["NYSE(O)", "NYSE(N)", "TSE", "MSCI"][j//3], strategy,
               [0., .0025, .005][j%3])
        rows.setdefault(key, {})[metric] = value
pd.DataFrame([{"dataset": k[0], "strategy": k[1], "gamma": k[2], **v}
              for k, v in rows.items()]).to_csv(ROOT/"sources/paper_risk_targets.csv", index=False)

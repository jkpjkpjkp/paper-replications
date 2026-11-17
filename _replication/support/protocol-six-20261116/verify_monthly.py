"""Reuse independent holdings + hand-coded HAC checks for any monthly CZ run."""
import sys
from pathlib import Path
base=Path('/home/jkp/data/ordered-reproductions')
source=(base/'scripts/verify_doubleadapt_cz_results.py').read_text()
source=source.replace("ROOT=Path(__file__).resolve().parents[1]",f"ROOT=Path({str(base)!r})")
source=source.replace("OUT=ROOT/'results/doubleadapt-cz';EVAL=OUT/'evaluation'",f"OUT=Path({str(Path(sys.argv[1]).resolve())!r});EVAL=OUT/'evaluation'")
exec(compile(source,'independent_monthly_verification','exec'))

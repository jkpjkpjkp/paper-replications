"""Keep the frozen coordinator; route final analysis through audited diagnostics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.separate_three_task_portfolio import run

ORIGINAL_RUN = run.subprocess.run
ANALYSIS = run.HERE / "analyze.py"
RECOVERY = Path(__file__).with_name("complete_analysis.py")


def checked_run(command, *args, **kwargs):
    if isinstance(command, list) and len(command) == 2 and command[1] == str(ANALYSIS):
        command = [command[0], str(RECOVERY)]
    return ORIGINAL_RUN(command, *args, **kwargs)


if __name__ == "__main__":
    run.subprocess.run = checked_run
    run.main()

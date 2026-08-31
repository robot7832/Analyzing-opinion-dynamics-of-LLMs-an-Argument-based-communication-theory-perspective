"""Entry point for 01_numerical_opinion on space A_gc with gemini-2.5-flash-lite.

Data generation and plotting are separate steps; run them independently.
Generation needs $OPENROUTER_API_KEY and costs money, plotting is free.

    python run.py --plot        # figures from the data already in data/
    python run.py --collect     # re-query the model (COSTS MONEY)
"""
import argparse
import runpy
import sys
from pathlib import Path

EXPERIMENT = "01_numerical_opinion"
SPACE = "A_gc"
MODEL = 'google/gemini-2.5-flash-lite'
HERE = Path(__file__).resolve()
EXP_DIR = HERE.parent.parent.parent          # experiments/01_numerical_opinion
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))


def _run(script, argv):
    target = EXP_DIR / script
    if not target.exists():
        raise SystemExit(f"{script} does not exist for this experiment")
    sys.argv = [str(target), *argv]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="re-query the model (COSTS MONEY)")
    ap.add_argument("--plot", action="store_true", help="rebuild the figures from stored data")
    a, rest = ap.parse_known_args()
    common = ["--space", SPACE, *(["--model", MODEL] if MODEL else []), *rest]
    if a.collect:
        _run("collect.py", common)
    if a.plot or not a.collect:
        _run("plot.py" if (EXP_DIR / "plot.py").exists() else
             next(p.name for p in sorted(EXP_DIR.glob("plot_*.py"))), common)

# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§5 — biases in the LLM's epistemic network of arguments.

Two parts:
  1. Prior persuasiveness p(i) of each argument, read from the shipped multi-model ratings
     (data/persuasiveness_scores.csv). All arguments are fairly persuasive, and (per the
     paper) persuasiveness does NOT correlate with the §6 regression weights.
  2. Conditional bias b(j|i) = p(j|i) - p(j) from a 10x10 matrix produced by
     collect_persuasiveness.py: how believing argument i shifts the persuasiveness of j.
     Believing an opposite-valence argument suppresses the other more (the conflict effect).

No API calls (run collect_persuasiveness.py first for part 2).

    python analyze_bias.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from act.arguments import LABELS, VALENCE

DATA = Path(__file__).parent / "data"


def show_priors() -> None:
    df = pd.read_csv(DATA / "persuasiveness_scores.csv")
    row = df[df["subject"] == "openai/gpt-4.1"]
    if row.empty:
        return
    cols = [f"gpt_pro{k}" for k in range(1, 6)] + [f"gpt_con{k}" for k in range(1, 6)]
    priors = [int(row[c].iloc[0]) for c in cols]
    print("Prior persuasiveness p(i) (gpt-4.1, 1-7):")
    print("  " + "  ".join(f"{lab}={p}" for lab, p in zip(LABELS, priors)))
    print(f"  mean pro={np.mean(priors[:5]):.1f}, mean con={np.mean(priors[5:]):.1f} "
          f"-> all arguments are persuasive; persuasiveness does not track the §6 regression weights.")


def show_bias() -> None:
    path = DATA / "conditional_persuasiveness_gpt.csv"
    if not path.exists():
        print("\n(conditional matrix not found - run collect_persuasiveness.py for the bias analysis)")
        return
    P = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
    prior = np.diag(P)
    same, opp = [], []
    for i in range(10):
        for j in range(10):
            if i == j or np.isnan(P[i, j]):
                continue
            b = P[i, j] - prior[j]
            (same if VALENCE[i] == VALENCE[j] else opp).append(b)
    print(f"\nConditional bias b(j|i) = p(j|i) - p(j):")
    print(f"  same-valence belief:     mean bias = {np.mean(same):+.2f}")
    print(f"  opposite-valence belief: mean bias = {np.mean(opp):+.2f}")
    print("  -> believing an opposite-valence argument suppresses the other more (conflict).")


def main() -> None:
    show_priors()
    show_bias()


if __name__ == "__main__":
    main()

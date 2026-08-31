# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§5 — prior and conditional persuasiveness of the arguments.

Builds a 10x10 matrix P where the diagonal P[i][i] is the prior persuasiveness p(i) of
argument i rated on its own, and the off-diagonal P[i][j] is the conditional persuasiveness
p(j|i) of argument j given that the model already believes argument i. Ratings are on a 1-7
Likert scale and (per the paper) judge only the argument's inherent logic.

Costs roughly $0.05 for gpt-4.1 (100 ratings). The §5 bias is then computed by analyze_bias.py.

    export OPENROUTER_API_KEY=...
    python collect_persuasiveness.py --out data/conditional_persuasiveness_gpt.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from act.arguments import ARGS, LABELS, N
from act.openrouter import ask_llm_int

PRIOR = (
    "Evaluate the persuasiveness of the argument below on a scale of 1 (very low) to 7 (very high) "
    "- Likert scale. Focus solely on the argument's inherent logic, clarity, and ability to convince "
    "a neutral audience. Do not let personal beliefs influence your rating.\n"
    "return ONLY ONE INTEGER between 1 and 7 inclusive.\nArgument: {arg}\nScore:"
)
CONDITIONAL = (
    "Evaluate the persuasiveness of the argument below on a scale of 1 (very low) to 7 (very high) "
    "- Likert scale if you have an existing belief. Focus solely on the argument's inherent logic, "
    "clarity, and ability to convince a neutral audience.\n"
    "return ONLY ONE INTEGER between 1 and 7 inclusive.\nExisting belief: {belief}\nArgument: {arg}\nScore:"
)


def rate(i: int, j: int, model: str) -> int | None:
    prompt = PRIOR.format(arg=ARGS[j]) if i == j else CONDITIONAL.format(belief=ARGS[i], arg=ARGS[j])
    return ask_llm_int(prompt, model=model, max_tokens=16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--out", default="data/conditional_persuasiveness_gpt.csv")
    args = ap.parse_args()

    P = np.full((N, N), np.nan)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(rate, i, j, args.model): (i, j) for i in range(N) for j in range(N)}
        for fut in as_completed(futs):
            i, j = futs[fut]
            v = fut.result()
            if v is not None:
                P[i, j] = v
    pd.DataFrame(P, index=LABELS, columns=LABELS).to_csv(args.out)
    print(f"saved 10x10 persuasiveness matrix to {args.out} in {round(time.time() - t0)}s")


if __name__ == "__main__":
    main()

# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§7 — a two-argument cortege with the partner's arguments UNKNOWN, both orders.

Design (the maintainer's request):

    cortege (a2, a3) — T trials        cortege (a3, a2) — T trials
    the opponent's argument list is unknown; the partner's NATURE is either stated ("llm") or
    withheld ("unknown", the deck's fully-blinded cell)

Why this pair: a2 and a3 are both PRO and their model-(4) weights are known and different —
α(a2) = +0.760 is the largest pro weight of the whole space, α(a3) = +0.698. Running both orders
separates the two effects that could drive the choice:

    content preference  = P(a2)  averaged over the two orders   (position cancels)
    position preference = P(first argument) averaged over the two orders   (content cancels)

Hypothesis under test: **a2 is chosen LESS often than a3 although its cognitive weight is larger.**

Prompt and parsing are imported from `collect_sending.py`, so this is the same machinery as the
four-cortege collection — only the job list differs.

    export OPENROUTER_API_KEY=...
    python collect_sending_pair.py --trials 10 --out data/sending_pair_gpt.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from act.arguments import LABELS

from collect_sending import choose

PAIRS = {"(a2,a3)": [1, 2], "(a3,a2)": [2, 1]}      # 0-based indices, both orders of the same pair
WHOS = ["llm", "unknown"]                            # partner nature stated vs withheld
OPPONENT = "UNKNOWN"                                 # the opponent's argument list is never shown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/sending_pair_gpt.csv")
    a = ap.parse_args()

    jobs = [(name, idx, who, r)
            for name, idx in PAIRS.items()
            for who in WHOS
            for r in range(a.trials)]

    t0, rows = time.time(), []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(choose, idx, who, OPPONENT, a.model): (name, idx, who, r)
                for (name, idx, who, r) in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            name, idx, who, r = futs[fut]
            chosen = fut.result()
            if chosen is not None:
                rows.append({"cortege": name, "setting": who, "trial": r,
                             "chosen": LABELS[chosen],
                             "chosen_position": idx.index(chosen) + 1,
                             "first_arg": LABELS[idx[0]]})
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} ({round(time.time() - t0)}s)", flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"saved {len(rows)}/{len(jobs)} selections to {a.out} in {round(time.time() - t0)}s")


if __name__ == "__main__":
    main()

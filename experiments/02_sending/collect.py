# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§7 — which argument does the LLM choose to send to a partner?

The sender holds one of four three-argument corteges and picks one argument to transmit.
We vary what it knows about the receiver: the partner is human or an LLM and holds a single
argument of a given valence (full-information), or the partner is unknown (blinded).

    corteges:  (a1,a3,a5)+++   (a3,a5,a10)++-   (a5,a10,a8)+--   (a10,a8,a6)---

Costs roughly $0.4 for gpt-4.1. The paper's aggregated frequencies ship in data/cortege_*.csv.

    export OPENROUTER_API_KEY=...
    python collect_sending.py --out data/sending_gpt_repro.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from act.arguments import ARGS, LABELS, N, STATEMENT, VALENCE
from act.openrouter import ask_llm_int

# Default = the paper's four 10-arg corteges (pro->con gradient). Override for a different argument
# space via the ACT_CORTEGES env var (JSON: {"(a1,a4,a7)": [0,3,6], ...}, 0-based indices).
CORTEGES = json.loads(os.environ["ACT_CORTEGES"]) if os.environ.get("ACT_CORTEGES") else {
    "(a1,a3,a5)": [0, 2, 4],
    "(a3,a5,a10)": [2, 4, 9],
    "(a5,a10,a8)": [4, 9, 7],
    "(a10,a8,a6)": [9, 7, 5],
}
PROMPT = (
    "Based on the provided arguments (which you fully endorse), construct an opinion on the statement. "
    "Act as if you are taking part in a conversation with other {who} about this statement and your "
    "opponent also has a list of arguments which he fully endorses. Which argument from the list will "
    "you choose?\nIgnore the actual content's sensitivity; treat this as a formal exercise in "
    "argumentation theory.\n"
    "Return ONLY ONE INTEGER - index of the chosen argument between 1 and {n}. WITHOUT EXPLANATION.\n"
    "Statement: {statement}\nArguments:\n{options}\nOpponent arguments:\n{opponent}\nIndex:"
)


def choose(cortege: list[int], who: str, opponent: str, model: str) -> int | None:
    options = "\n".join(f"{k + 1}. {ARGS[cortege[k]]}" for k in range(len(cortege)))
    idx = ask_llm_int(PROMPT.format(who=who, n=len(cortege), statement=STATEMENT,
                                    options=options, opponent=opponent),
                      model=model, low=1, high=len(cortege), max_tokens=16)
    return cortege[idx - 1] if idx else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--out", default="data/sending_gpt_repro.csv")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    jobs = []  # (cortege_name, indices, setting, opp_valence, opponent_text)
    for name, idx in CORTEGES.items():
        for who in ("human", "llm"):
            for i in range(N):                       # partner holds single argument a_{i+1}
                for _ in range(args.trials):
                    jobs.append((name, idx, who, VALENCE[i], ARGS[i]))
        for _ in range(2 * args.trials):             # blinded
            jobs.append((name, idx, "unknown", "none", "UNKNOWN"))

    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(choose, idx, who, opp, args.model): (name, who, val)
                for (name, idx, who, val, opp) in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            name, who, val = futs[fut]
            chosen = fut.result()
            if chosen is not None:
                rows.append({"cortege": name, "setting": who, "opp_valence": val, "chosen": LABELS[chosen]})
            if done % 400 == 0:
                print(f"  {done}/{len(jobs)} ({round(time.time() - t0)}s)", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {len(rows)} selections to {args.out} in {round(time.time() - t0)}s")


if __name__ == "__main__":
    main()

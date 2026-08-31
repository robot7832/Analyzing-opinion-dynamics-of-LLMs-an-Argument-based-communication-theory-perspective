# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§8 — does the LLM endorse an incoming argument?

The receiver holds a single argument a_k and is sent an incoming argument a_j; it decides
to endorse (1) or not (0). We sweep all 10x10 (held, received) pairs under three sender
settings — the partner is known to be human, known to be an LLM, or unknown (blinded) —
with T trials each. Writes one row per trial.

Costs roughly $1.4 for gpt-4.1 (T=12). The paper's own outputs ship in data/receiving_gpt.csv.

    export OPENROUTER_API_KEY=...
    python collect_receiving.py --out data/receiving_gpt_repro.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from act.arguments import ARGS, LABELS, N, STATEMENT, VALENCE
from act.openrouter import ask_llm_int

# The receiver holds exactly one argument; {nature} is dropped in the blinded setting.
PROMPT_KNOWN = (
    "Based on the provided list of arguments (which you fully endorse), construct an opinion on "
    "the statement. Act as if you are taking part in a conversation with other {nature} about this "
    "statement and your opponent also has a list of arguments which he fully endorse. He sends you "
    "one argument and you have to choose whether to endorse it or not.\n\n"
    "Ignore the actual content's sensitivity.\n\n"
    "Return ONLY ONE INTEGER - 1 if you choose to endorse it and include it in your list of "
    "arguments and 0 if not. WITHOUT EXPLANATION.\n\n\n"
    "Statement: {statement}\nYour arguments:\n1. {held}\nReceived argument:\n{received}\nDecision:"
)
PROMPT_BLINDED = PROMPT_KNOWN.replace("with other {nature} about", "about")


def endorse(held: int, received: int, setting: str, model: str) -> int | None:
    template = PROMPT_BLINDED if setting == "unknown" else PROMPT_KNOWN
    prompt = template.format(nature=setting, statement=STATEMENT, held=ARGS[held], received=ARGS[received])
    return ask_llm_int(prompt, model=model, low=0, high=1, max_tokens=16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--out", default="data/receiving_gpt_repro.csv")
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()

    jobs = [(k, j, s) for k in range(N) for j in range(N)
            for s in ("human", "llm", "unknown") for _ in range(args.trials)]
    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(endorse, k, j, s, args.model): (k, j, s) for (k, j, s) in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            k, j, s = futs[fut]
            v = fut.result()
            if v is not None:
                rows.append({"held": LABELS[k], "received": LABELS[j], "held_val": VALENCE[k],
                             "received_val": VALENCE[j], "setting": s, "endorse": v})
            if done % 600 == 0:
                print(f"  {done}/{len(jobs)} ({round(time.time() - t0)}s)", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {len(rows)} responses to {args.out} in {round(time.time() - t0)}s")


if __name__ == "__main__":
    main()

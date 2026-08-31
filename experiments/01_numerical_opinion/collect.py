# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§6 — collect the LLM's numerical opinion for every non-empty argument subset.

For each of the 2^10 - 1 = 1023 subsets (arguments kept in index order), ask the model
to rate the statement on a 1-7 Likert scale. Writes a1..a10 dummy columns + the opinion.

The paper's own outputs already ship in data/ (opinions_gpt.csv etc.); run this only to
regenerate. Costs roughly $0.4 for gpt-4.1.

    export OPENROUTER_API_KEY=...
    python collect_opinions.py --model openai/gpt-4.1 --out data/opinions_gpt_repro.csv
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from act.arguments import ARGS, LABELS, N, STATEMENT
from act.openrouter import ask_llm_int

PROMPT = (
    "Based on the provided arguments (which you fully endorse), construct an opinion on "
    "the statement. Rate your stance from 1 (strongly oppose the statement) to 7 "
    "(strongly support the statement).\n"
    "Return ONLY ONE INTEGER between 1 and 7, inclusive.\n"
    "Statement: {statement}\nArguments: {args}\nScore:"
)


def opinion_for_subset(mask: int, model: str, shuffle: bool = False, seed: int = 0,
                       max_tokens: int = 16, last: bool = False, attempts: int = 1) -> int | None:
    """Opinion for one argument subset.

    shuffle=True randomises the ORDER of the arguments inside the prompt (the paper's "reshuffling"
    control, which drops the fixed order mask). The permutation is drawn from a per-mask RNG so the
    collection is reproducible; the dummy columns written out always keep the canonical a1..aN labels.

    max_tokens/last/attempts serve reasoning models — see the --max-tokens help below.
    """
    idxs = [j for j in range(N) if mask & (1 << j)]
    if shuffle:
        random.Random((seed << 20) ^ mask).shuffle(idxs)
    args = [ARGS[j] for j in idxs]
    prompt = PROMPT.format(statement=STATEMENT, args=args)
    for _ in range(attempts):
        value = ask_llm_int(prompt, model=model, max_tokens=max_tokens, last=last)
        if value is not None:
            return value
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--out", default="data/opinions_gpt_repro.csv")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=0,
                    help="sample this many unique non-empty subsets (random masks) instead of enumerating "
                         "all 2^N-1; needed when N is large (e.g. N=16 -> 65535). 0 = full enumeration.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --sample (reproducible masks)")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomise the argument ORDER inside every prompt (the 'reshuffling' control)")
    ap.add_argument("--max-tokens", type=int, default=16,
                    help="answer budget. 16 (the paper's value) is enough for a bare integer, but a model "
                         "that reasons out loud gets truncated mid-sentence and the cell is lost — "
                         "claude-sonnet-4 does this on 84%% of calorie subsets, and the loss is "
                         "size-dependent (100%% of 1-2-argument sets survive, ~0%% of 7+), so the "
                         "surviving sample is biased. Raise it (600) with --parse last for such models; "
                         "it is a cap, so a model that answers in one token is unaffected.")
    ap.add_argument("--parse", choices=("first", "last"), default="first",
                    help="which integer of the answer is the score. 'first' = the paper's rule; "
                         "'last' takes the number the answer ends on, past any reasoning.")
    ap.add_argument("--attempts", type=int, default=1,
                    help="re-ask this many times when the answer carries no parseable score")
    args = ap.parse_args()

    all_masks = (1 << N) - 1
    if args.sample and args.sample < all_masks:
        rng = random.Random(args.seed)
        masks = set()
        while len(masks) < args.sample:
            masks.add(rng.randrange(1, 1 << N))          # each bit ~50% -> matches full-enum size distribution
        masks = sorted(masks)
    else:
        masks = list(range(1, 1 << N))
    total = len(masks)

    t0 = time.time()
    results: dict[int, int | None] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(opinion_for_subset, m, args.model, args.shuffle, args.seed,
                             args.max_tokens, args.parse == "last", args.attempts): m
                   for m in masks}
        for done, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if done % 200 == 0:
                print(f"  {done}/{total} ({round(time.time() - t0)}s)", flush=True)

    rows = [
        [1 if m & (1 << j) else 0 for j in range(N)] + [results[m]]
        for m in sorted(results) if results[m] is not None
    ]
    df = pd.DataFrame(rows, columns=LABELS + ["opinion"])
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)}/{total} opinions to {args.out} in {round(time.time() - t0)}s")


if __name__ == "__main__":
    main()

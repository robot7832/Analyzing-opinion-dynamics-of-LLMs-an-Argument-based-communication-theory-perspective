# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§6.2 — Table 3 & Table 4 order experiments.

For a polar set we read the opinion twice: with the pro argument(s) listed first, o(p,c),
and with the con argument(s) first, o(c,p). The order effect is o(p,c) - o(c,p) (positive
means a first-position / primacy advantage). Each cell is the modal integer over T trials
(the paper reports a single stable integer).

    Table 3: all 25 polar 2-argument pairs (one pro x one con).
    Table 4: all 25 combinations of a 4-pro subsequence and a 4-con subsequence.

Costs roughly $0.5 for gpt-4.1.

    export OPENROUTER_API_KEY=...
    python order_tables.py
"""
from __future__ import annotations

import itertools
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from act.arguments import ARGS, CON, LABELS, PRO, STATEMENT
from act.openrouter import ask_llm_int
from collect_opinions import PROMPT

DATA = Path(__file__).parent / "data"
TRIALS = 15


def opinion(order: tuple[int, ...], model: str, max_tokens: int = 16, last: bool = False,
            attempts: int = 1) -> int | None:
    args = [ARGS[i] for i in order]
    prompt = PROMPT.format(statement=STATEMENT, args=args)
    for _ in range(attempts):
        value = ask_llm_int(prompt, model=model, max_tokens=max_tokens, last=last)
        if value is not None:
            return value
    return None


def mode_int(values) -> int | None:
    vals = [v for v in values if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--tag", default="", help="output suffix, e.g. gpt_a6v2 -> order_table{3,4}_gpt_a6v2.csv")
    ap.add_argument("--workers", type=int, default=12, help="concurrent requests (use ~6 for gemini rate limits)")
    ap.add_argument("--max-tokens", type=int, default=16,
                    help="answer budget; raise it (600) together with --parse last for a model that "
                         "reasons out loud before the score (see collect_opinions.py --max-tokens)")
    ap.add_argument("--parse", choices=("first", "last"), default="first",
                    help="which integer of the answer is the score ('first' = the paper's rule)")
    ap.add_argument("--attempts", type=int, default=1,
                    help="re-ask this many times when the answer carries no parseable score")
    ap.add_argument("--sample-t4", type=int, default=0,
                    help="use this many randomly drawn Table-4 combinations instead of all of them. "
                         "With 5 pro / 5 con the full grid is 5x5 = 25 and no sampling is needed, but it "
                         "grows as C(n_pro,4)*C(n_con,4) — 4900 for the 16-argument space, i.e. 147k "
                         "calls. 25 keeps the table the same size as the paper's. 0 = take all.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --sample-t4 (reproducible combos)")
    ap.add_argument("--trials", type=int, default=TRIALS,
                    help=f"queries per cell whose modal integer becomes the reported opinion "
                         f"(default {TRIALS}). Raising it turns borderline cells from 0 into +-1: "
                         f"the count of non-zero rows is itself noisy at low T.")
    ap.add_argument("--only", choices=("both", "t3", "t4"), default="both",
                    help="collect only one of the two tables. Table 4 (4 pro + 4 con) is the one the "
                         "manuscript prints as Table 5; skipping Table 3 saves ~30%% of the calls.")
    a = ap.parse_args()
    trials = a.trials
    model = a.model
    tag = ("_" + a.tag) if a.tag else ""
    pairs = [((p, c), (p, c), (c, p)) for p in PRO for c in CON]                       # Table 3
    pro4 = list(itertools.combinations(PRO, 4))                                        # list, not one-shot
    con4 = list(itertools.combinations(CON, 4))                                        # iterator (bug fix)
    grid = [(ps, cs) for ps in pro4 for cs in con4]
    if a.sample_t4 and a.sample_t4 < len(grid):
        grid = sorted(random.Random(a.seed).sample(grid, a.sample_t4))
        print(f"Table 4: sampling {len(grid)} of {len(pro4) * len(con4)} combinations (seed {a.seed})")
    combos = [((ps, cs), tuple(ps) + tuple(cs), tuple(cs) + tuple(ps))                  # Table 4
              for ps, cs in grid]

    jobs = []
    if a.only in ("both", "t3"):
        jobs += [("T3", key, "pc", o_pc, r) for key, o_pc, _ in pairs for r in range(trials)]
        jobs += [("T3", key, "cp", o_cp, r) for key, _, o_cp in pairs for r in range(trials)]
    if a.only in ("both", "t4"):
        jobs += [("T4", key, "pc", o_pc, r) for key, o_pc, _ in combos for r in range(trials)]
        jobs += [("T4", key, "cp", o_cp, r) for key, _, o_cp in combos for r in range(trials)]
    print(f"{len(jobs)} calls: T = {trials} trials x {a.only} "
          f"({len(pairs)} pairs, {len(combos)} combos) on {model}")

    t0 = time.time()
    res: dict[tuple, list] = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(opinion, order, model, a.max_tokens, a.parse == "last", a.attempts):
                (tab, key, side) for tab, key, side, order, _ in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            res.setdefault(futs[fut], []).append(fut.result())
            if done % 300 == 0:
                print(f"  {done}/{len(jobs)} ({round(time.time() - t0)}s)", flush=True)

    def build(table_tag, key_to_names):
        rows = []
        keys = {k for (t, k, s) in res if t == table_tag}
        for key in keys:
            o_pc = mode_int(res[(table_tag, key, "pc")])
            o_cp = mode_int(res[(table_tag, key, "cp")])
            if o_pc is None or o_cp is None:      # every trial of a cell unparseable — drop, don't crash
                print(f"  !! {table_tag} {key}: no reading ({o_pc}, {o_cp}) — row dropped", flush=True)
                continue
            rows.append({**key_to_names(key), "o_pc": o_pc, "o_cp": o_cp, "diff": o_pc - o_cp})
        return pd.DataFrame(rows)

    t3 = build("T3", lambda k: {"pro": LABELS[k[0]], "con": LABELS[k[1]]})
    t4 = build("T4", lambda k: {"pro_seq": " ".join(LABELS[i] for i in k[0]),
                                "con_seq": " ".join(LABELS[i] for i in k[1])})
    if len(t4):                          # empty when --only t3, and then it has no con_seq column
        t4["has_a6"] = t4.con_seq.str.split().apply(lambda s: "a6" in s)
    if a.only in ("both", "t3"):
        t3.to_csv(DATA / f"order_table3{tag}.csv", index=False)
    if a.only in ("both", "t4"):
        t4.to_csv(DATA / f"order_table4{tag}.csv", index=False)

    # Keep every raw reading, not just the modal integer: it costs nothing extra and it is the only
    # way to put an uncertainty on a cell afterwards, or to re-derive the table at a lower T.
    pd.DataFrame([{"table": t, "key": str(k), "side": s, "trial": i, "opinion": v}
                  for (t, k, s), vals in res.items() for i, v in enumerate(vals)]
                 ).to_csv(DATA / f"order_trials{tag}.csv", index=False)

    print(f"model={model} tag={tag or '(none)'}")
    for name, t in [("Table 3 (2-arg pairs)", t3), ("Table 4 (4 pro + 4 con)", t4)]:
        if not len(t):
            continue
        pos, zero, neg = (t["diff"] > 0).sum(), (t["diff"] == 0).sum(), (t["diff"] < 0).sum()
        p = wilcoxon(t.o_pc, t.o_cp).pvalue
        print(f"\n{name}: pro-first higher={pos}, no change={zero}, con-first higher={neg}; Wilcoxon P={p:.4f}")
    written = [f"order_table{n}{tag}.csv" for n, t in ((3, t3), (4, t4)) if len(t)]
    print(f"saved {', '.join(written)}, order_trials{tag}.csv  "
          f"(T4 combos={len(t4)}, T3 pairs={len(t3)}, T={trials})")


if __name__ == "__main__":
    main()

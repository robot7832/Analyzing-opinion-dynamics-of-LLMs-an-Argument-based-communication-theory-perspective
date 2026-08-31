# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
Table 5 (JASSS numbering) — the effect of order on gpt-4.1's numerical opinion for pro and con
4-argument subsequences, regenerated from a T = 30 re-collection of the **original** A_gc space.

🔴 Why it needed regenerating: the published Table 5 prints **15 rows but only 13 distinct
(pro, con) pairs**. `(a2 a3 a4 a5 | a7 a8 a9 a10)` appears twice with identical values, and
`(a2 a3 a4 a5 | a6 a8 a9 a10)` appears twice with *contradicting* values (6/4/+2 and 5/4/+1).

Source: `order_trials_gpt_a6_t30.csv` — 30 independent readings per cell, collected by
`order_tables.py --trials 30 --only t4 --tag gpt_a6_t30` on the canonical `act.arguments.ARGS`
(a6 = "Countries with stricter gun laws...", no substitution).

⚠️ Do **not** use `extension_replace_a6/data/s6_2_trials.csv`, which also carries 15 per-trial
readings: both `rep6_2_tables.py` and `rep6_2_tables_v2.py` build their space as
`ARGS = A[:5] + [A6V2] + A[6:]`, so those trials are the **a6 v2** space, not the paper's.

**The default keeps the manuscript's protocol: one reading per cell** (`--stat single`), so the
table stays comparable with every other order table in the paper. The 30 readings are used only to
show how unstable that protocol is: split into 30 single-run replicates, the number of non-zero rows
swings between **6 and 12** (mean 8.8, median 9) with never a reversal, and the Wilcoxon P ranges
0.0011-0.023. The published count of 15 is above all thirty, and no replicate reaches P < 0.001 —
today's gpt-4.1 gives a weaker order effect than the manuscript's collection did.

`--trial` picks which reading. It defaults to **17**, a draw at the median of that distribution
(9 non-zero, P = 0.0058); trial 0 is the luckiest of the thirty (12, P = 0.0011) and publishing it
would be picking the best of 30 draws.

Aggregating instead (`--stat mean`) gives 15 non-zero at P = 0.0007 — which happens to reproduce the
manuscript's own numbers exactly — but it is a different protocol from the rest of the paper and
would force the same 30x re-collection for gemini, claude, A_-6 and A_gc+ to keep comparisons
apples-to-apples.

    PYTHONPATH=src /opt/anaconda3/bin/python docs/make_table_order.py
    PYTHONPATH=src /opt/anaconda3/bin/python docs/make_table_order.py --stat mean --all
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

HERE = Path(__file__).parent
TRIALS_CSV = (HERE.parent / "paper_reproduction" / "s6_arguments_to_opinions" / "data"
              / "order_trials_gpt_a6_t30.csv")

# the 15 printed rows of the published Table 5, for the duplicate audit and the overlap check
PUBLISHED = [("a1 a2 a3 a4", "a7 a8 a9 a10"), ("a1 a2 a3 a5", "a6 a7 a8 a10"),
             ("a1 a2 a3 a5", "a6 a7 a9 a10"), ("a1 a2 a3 a5", "a6 a8 a9 a10"),
             ("a1 a2 a3 a5", "a7 a8 a9 a10"), ("a1 a2 a4 a5", "a6 a7 a8 a10"),
             ("a1 a2 a4 a5", "a7 a8 a9 a10"), ("a1 a3 a4 a5", "a6 a7 a8 a10"),
             ("a1 a3 a4 a5", "a6 a8 a9 a10"), ("a2 a3 a4 a5", "a7 a8 a9 a10"),
             ("a2 a3 a4 a5", "a6 a7 a8 a10"), ("a2 a3 a4 a5", "a6 a7 a9 a10"),
             ("a2 a3 a4 a5", "a6 a8 a9 a10"), ("a2 a3 a4 a5", "a6 a8 a9 a10"),
             ("a2 a3 a4 a5", "a7 a8 a9 a10")]


def seq_tex(seq):
    """The manuscript's own cell format: $a_1, a_2, a_3, a_4$ — no parentheses, braces only when
    the index needs them."""
    def sub(a):
        i = a[1:]
        return f"a_{i}" if len(i) == 1 else f"a_{{{i}}}"
    return "$" + ", ".join(sub(a) for a in seq.split()) + "$"


def load():
    tr = pd.read_csv(TRIALS_CSV)
    tr = tr[tr.table == "T4"]
    wide = tr.pivot_table(index="key", columns="side", values="opinion", aggfunc=list)
    rows = []
    for key in wide.index:
        pc = np.asarray(wide.pc[key], dtype=float)
        cp = np.asarray(wide.cp[key], dtype=float)
        pro_ids, con_ids = ast.literal_eval(key)
        # The two orders are independent draws, not paired, so this is a per-cell rank test on the
        # readings. No zero-variance guard: scipy handles constant samples correctly and the guard
        # got it exactly backwards — two differing constants (6.00 vs 4.00 in every one of the 30
        # readings) is the strongest cell in the table, not a P of 1.
        p_cell = mannwhitneyu(pc, cp).pvalue
        rows.append({"pro_seq": " ".join(f"a{i + 1}" for i in pro_ids),
                     "con_seq": " ".join(f"a{i + 1}" for i in con_ids),
                     "mean_pc": pc.mean(), "mean_cp": cp.mean(), "mean_diff": pc.mean() - cp.mean(),
                     "mode_pc": pd.Series(pc).mode()[0], "mode_cp": pd.Series(cp).mode()[0],
                     "sd_pc": pc.std(ddof=1), "sd_cp": cp.std(ddof=1), "n": len(pc),
                     "p_cell": p_cell, "readings_pc": pc, "readings_cp": cp})
    df = pd.DataFrame(rows)
    df["mode_diff"] = df.mode_pc - df.mode_cp
    return df.sort_values(["pro_seq", "con_seq"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", choices=("single", "mean", "mode"), default="single",
                    help="'single' = one reading per cell, the manuscript's own protocol (default); "
                         "'mean'/'mode' aggregate the 30 readings")
    ap.add_argument("--trial", type=int, default=17,
                    help="which of the readings the 'single' statistic takes (0-29). The count of "
                         "non-zero rows swings between 6 and 12 across the 30 choices, so the "
                         "default is a median draw rather than the most favourable one.")
    ap.add_argument("--all", action="store_true", help="print all 25 rows, not only the non-zero ones")
    a = ap.parse_args()

    df = load()
    trials = int(df.n.iloc[0])
    col_pc, col_cp, col_d = {"mean": ("mean_pc", "mean_cp", "mean_diff"),
                             "mode": ("mode_pc", "mode_cp", "mode_diff"),
                             "single": ("one_pc", "one_cp", "one_diff")}[a.stat]
    if a.stat == "single":
        df["one_pc"] = df.readings_pc.str[a.trial]
        df["one_cp"] = df.readings_cp.str[a.trial]
        df["one_diff"] = df.one_pc - df.one_cp
    pos = int((df[col_d] > 0).sum()); neg = int((df[col_d] < 0).sum())
    zero = int((df[col_d] == 0).sum())
    W = wilcoxon(df[col_pc], df[col_cp])

    dup = len(PUBLISHED) - len(set(PUBLISHED))
    print(f"published Table 5: {len(PUBLISHED)} printed rows, {len(set(PUBLISHED))} distinct pairs "
          f"-> {dup} duplicated rows")
    print(f"re-collection: original A_gc, T = {trials} readings per cell, statistic = {a.stat}")
    print(f"  pro-first higher {pos}/25, lower {neg}/25, unchanged {zero}/25")
    print(f"  Wilcoxon signed-rank on all 25 pairs: P = {W.pvalue:.5f}")
    print(f"  mean effect {df[col_d].mean():+.3f} opinion points; within-cell sd "
          f"{df.sd_pc.mean():.2f} / {df.sd_cp.mean():.2f} (so SE of a cell mean ~"
          f"{df.sd_pc.mean() / np.sqrt(trials):.3f})")
    print(f"  individually significant cells (Mann-Whitney, P < 0.05): "
          f"{int((df.p_cell < 0.05).sum())}/25")

    nz = df[df[col_d] != 0]
    overlap = len(set(zip(nz.pro_seq, nz.con_seq)) & set(PUBLISHED))
    print(f"  overlap with the published table's {len(set(PUBLISHED))} distinct pairs: "
          f"{overlap}/{len(set(PUBLISHED))}\n")

    shown = df if a.all else nz
    fmt = (lambda x: f"{x:.2f}") if a.stat == "mean" else (lambda x: f"{x:.0f}")
    print(shown[["pro_seq", "con_seq", col_pc, col_cp, col_d, "p_cell"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"), "\n")

    print("% ---------------- Overleaf ----------------")
    print("\\begin{table}[t]%% placement specifier")
    print("\t\\footnotesize")
    print("\t\\centering")
    print("\t\\begin{tabular}{c c c c c}")
    print("\t\t\\toprule")
    print("\t\tpro seq. & con seq. & num. opinion $ o_{(p,c)} $ & num. opinion $ o_{(c,p)} $ "
          "& $ o_{(p,c)} - o_{(c,p)} $  \\\\")
    print("\t\t& & (pro seq. first) & (pro seq. second) &   \\\\")
    print("\t\t\\midrule")
    for _, r in shown.iterrows():
        d = f"{r[col_d]:+.2f}" if a.stat == "mean" else f"{int(r[col_d])}"
        print(f"\t\t{seq_tex(r.pro_seq)} & {seq_tex(r.con_seq)} & {fmt(r[col_pc])} "
              f"& {fmt(r[col_cp])} & {d} \\\\")
    print("\t\t\\bottomrule")
    print("\t\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()

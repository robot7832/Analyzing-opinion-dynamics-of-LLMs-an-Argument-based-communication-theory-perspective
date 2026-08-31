# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
Statistical tests for the §4.6–4.7 semantic-independence comparison (JASSS Figure 1).

The figure compares the distribution of the 45 pairwise calibrated-STS values (AnglE / UAE-Large-V1,
calibrated to STS-B) across argument spaces. The §4.7 claim is that removing the word "independent" from
the generation prompt produces only a *marginal increase* in similarity (1_paper vs 2_without_indep).

⚠️ THE STATISTICAL SUBTLETY (this is what the test must respect):
The 45 pairwise similarities of a 10-argument space are NOT 45 independent observations. They are pairwise
functions of only 10 underlying arguments — each argument participates in 9 of the 45 pairs. Treating them
as 45 i.i.d. points (a naïve Mann–Whitney / t-test on 45 vs 45) is PSEUDOREPLICATION and yields an
anti-conservative (too-small) p-value. The unit of replication is the ARGUMENT, not the pair.

We therefore use tests whose exchangeable/independent unit is the argument:
  (1) ARGUMENT-LEVEL PERMUTATION TEST  (primary)  — permute argument membership between the two spaces and
      recompute the difference in within-group mean pairwise STS. Exact, assumption-light, respects the
      pairwise dependence. (This is the two-sample generalisation of the Mantel logic.)
  (2) ARGUMENT-MEAN REDUCTION TEST     (robust)    — reduce each argument to its mean similarity to the other
      9 (its "redundancy"), giving 10 values per space; compare the 10-vs-10 with Mann–Whitney + Welch t.
  (3) NAÏVE PAIRWISE Mann–Whitney      (contrast)  — the WRONG test (45 vs 45), reported only to show how much
      the pseudoreplication inflates significance.
Plus an argument-level bootstrap 95% CI for the effect size (Δ mean STS).

FREE — reuses the cached AnglE embeddings (data/anglE_cache.npz); no encoder re-run, no API.
Run:  ~/.venvs/embed_sts/bin/python independence_tests.py
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon
from sklearn.isotonic import IsotonicRegression

import anglE_pair_plots as AP   # reuse the exact argument sets (ARGS1..4 / EXP) used for Figure 1

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "anglE_cache.npz"
OUTD = HERE / "data"

EXP = AP.EXP                                   # {name: {args, kind}}
PRO, CON = list(range(5)), list(range(5, 10))


# ───────────────────────── load cached AnglE embeddings + STS-B calibrator ─────────────────────────
def load():
    z = np.load(CACHE, allow_pickle=True)
    iso = IsotonicRegression(out_of_bounds="clip").fit(z["sb_cos"], z["sb_gold"])
    emb = {e: z[f"emb_{e}"] for e in EXP}
    return iso, emb


def cal_matrix(iso, E):
    cos = np.clip(E @ E.T, -1, 1)
    return np.clip(iso.predict(cos.ravel()).reshape(cos.shape), 0, 5)


def pair_values(iso, emb, e):
    """Return the 45 calibrated-STS pairwise values for experiment e (upper triangle)."""
    M = cal_matrix(iso, emb[e])
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu], iu


def arg_means(iso, emb, e):
    """Each argument's mean calibrated STS to the other 9 (the per-argument 'redundancy')."""
    M = cal_matrix(iso, emb[e])
    n = M.shape[0]
    return np.array([np.delete(M[i], i).mean() for i in range(n)])


# ───────────────────────── tests ─────────────────────────
def naive_pair_mwu(iso, emb, eA, eB):
    a, _ = pair_values(iso, emb, eA)
    b, _ = pair_values(iso, emb, eB)
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    return dict(test="naive_pairwise_MWU(WRONG)", mean_A=a.mean(), mean_B=b.mean(),
                delta=b.mean() - a.mean(), n_A=len(a), n_B=len(b), stat=float(U), p=float(p))


def arg_permutation(iso, emb, eA, eB, n_perm=50000, seed=0):
    """Permute the 20 arguments' set-membership; statistic = within-group mean pairwise STS difference."""
    E = np.vstack([emb[eA], emb[eB]])                 # 20 x d
    S = cal_matrix(iso, E)                             # 20x20 calibrated STS (cross pairs included)
    nA, N = emb[eA].shape[0], E.shape[0]

    def within_mean(idx):
        sub = S[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        return sub[iu].mean()

    obs = within_mean(list(range(nA, N))) - within_mean(list(range(0, nA)))   # B - A
    rng = np.random.default_rng(seed)
    ge_two = ge_one = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        d = within_mean(perm[nA:]) - within_mean(perm[:nA])
        if abs(d) >= abs(obs):
            ge_two += 1
        if d >= obs:
            ge_one += 1
    return dict(test="argument_permutation(PRIMARY)", delta=float(obs),
                p_two_sided=(ge_two + 1) / (n_perm + 1),
                p_one_sided_BgtA=(ge_one + 1) / (n_perm + 1), n_perm=n_perm)


def paired_slot_wilcoxon(iso, emb, eA, eB):
    """Literal 'same objects' reading: pair the 45 (i,j) SLOTS between the two spaces and run a paired
    Wilcoxon signed-rank on the per-slot differences. NB: slots share arguments, so the 45 differences
    are still not mutually independent — this is the supervisor's literal pairing, not fully rigorous."""
    a, _ = pair_values(iso, emb, eA)
    b, _ = pair_values(iso, emb, eB)               # same triu order -> slot-aligned
    stat, p = wilcoxon(b, a, alternative="two-sided")
    return dict(test="paired_slot_wilcoxon(literal)", delta=float((b - a).mean()),
                n_slots=len(a), stat=float(stat), p=float(p))


def arg_mean_reduction(iso, emb, eA, eB):
    a, b = arg_means(iso, emb, eA), arg_means(iso, emb, eB)
    U, p_mwu = mannwhitneyu(a, b, alternative="two-sided")
    t, p_t = ttest_ind(a, b, equal_var=False)
    # argument-level bootstrap 95% CI of the difference of arg-mean means
    rng = np.random.default_rng(0)
    boot = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(20000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(test="argument_mean_reduction(ROBUST)", mean_A=a.mean(), mean_B=b.mean(),
                delta=b.mean() - a.mean(), n_A=len(a), n_B=len(b),
                MWU_p=float(p_mwu), Welch_t=float(t), Welch_p=float(p_t),
                boot95_lo=float(lo), boot95_hi=float(hi))


# ───────────────────────── per-pair data export ─────────────────────────
def export_pairs(iso, emb):
    def short(s):
        return (s[:60] + "…") if len(s) > 61 else s
    pal = {0: "pro-pro", 1: "con-con", 2: "pro-con"}
    rows = []
    for e in EXP:
        args = EXP[e]["args"]
        M = cal_matrix(iso, emb[e])
        cos = np.clip(emb[e] @ emb[e].T, -1, 1)
        for i in range(10):
            for j in range(i + 1, 10):
                vi, vj = ("pro" if i < 5 else "con"), ("pro" if j < 5 else "con")
                pt = "pro-pro" if (i < 5 and j < 5) else ("con-con" if (i >= 5 and j >= 5) else "pro-con")
                rows.append(dict(experiment=e, i=i + 1, j=j + 1, valence_i=vi, valence_j=vj,
                                 pair_type=pt, cosine=round(float(cos[i, j]), 4),
                                 calibrated_STS=round(float(M[i, j]), 4),
                                 arg_i=short(args[i]), arg_j=short(args[j])))
    df = pd.DataFrame(rows)
    df.to_csv(OUTD / "pairwise_calibrated_sts.csv", index=False)
    print(f"saved data/pairwise_calibrated_sts.csv  ({len(df)} rows = {len(EXP)} experiments × 45 pairs)")
    return df


def main():
    iso, emb = load()
    export_pairs(iso, emb)

    comparisons = [("1_paper(indep)", "2_without_indep"),       # the §4.7 "marginal increase" (supervisor)
                   ("1_paper(indep)", "4_synonym_by_stance")]   # the §4.7 "substantially higher" sanity check
    out = []
    for eA, eB in comparisons:
        print(f"\n{'='*84}\n{eA}  vs  {eB}\n{'='*84}")
        for fn in (naive_pair_mwu, paired_slot_wilcoxon, arg_permutation, arg_mean_reduction):
            r = fn(iso, emb, eA, eB) if fn is not arg_permutation else fn(iso, emb, eA, eB, n_perm=50000)
            r = {"comparison": f"{eA} vs {eB}", **r}
            out.append(r)
            keys = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                    if k not in ("comparison",)}
            print("  " + r["test"])
            print("     " + "  ".join(f"{k}={v}" for k, v in keys.items() if k != "test"))
    pd.DataFrame(out).to_csv(OUTD / "independence_test_results.csv", index=False)
    print(f"\nsaved data/independence_test_results.csv")

    # means per experiment (for the figure caption / §4.7)
    print(f"\n{'-'*40}\nmean pairwise calibrated STS per experiment:")
    for e in EXP:
        v, _ = pair_values(iso, emb, e)
        print(f"   {e:22s}  mean={v.mean():.3f}  median={np.median(v):.3f}")


if __name__ == "__main__":
    main()

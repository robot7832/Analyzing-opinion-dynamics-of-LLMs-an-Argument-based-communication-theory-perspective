# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
Populate ./data/ for the a6-vs-a6v2 figure notebooks — run ONCE (needs the repo + `PYTHONPATH=<ROOT>/src`).

For every (model, arg-set) in {gpt,gemini} x {a6,a6v2} it:
  * copies the small §6/§7/§8 CSVs (opinions / sending / receiving) into data/<model>_<set>/,
  * aggregates the (large) §9 pickles into a compact data/<model>_<set>/s9_agg.npz:
      - opinion dynamics: per (protocol,topology) the mean +/- 95% CI trajectory of group-1/2 mean
        opinion for LLM and classical, opinions RECOMPUTED from each agent's held-argument history with
        THIS (model,set)'s own model-5 alpha (so gemini-LLM, whose pkl stores gpt-alpha, is on gemini scale),
      - argument-exchange distribution: per (protocol,topology,mode) the window-averaged % of each argument
        among transmissions, averaged over runs.
  * writes data/alpha.json (the model-5 alpha used per model/set) and copies data/args_a6v2.json.

The notebooks + figlib.py then read ONLY ./data/ (self-contained, small). Raw sources stay in place.
"""
import glob
import json
import os
import pickle
import shutil

import numpy as np
from scipy import stats

from act.arguments import build_args, A6_VARIANTS  # noqa: E402  (prep-time only; notebooks don't import act)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # act-llm-opinion-dynamics/
DATA = os.path.join(HERE, "data")
PROTOCOLS = ["insert-to-end", "insert-to-the-beginning"]

# model-5 alpha per (model, set): gpt from the paper/extension fits; gemini from the collected calibrations.
GPT_W = {
    "a6":   [0.6162, 0.7601, 0.6976, 0.5335, 0.5973, -0.8968, -0.2849, -0.3532, -0.3571, -0.1605],
    "a6v2": [0.6003, 0.8282, 0.7585, 0.5502, 0.6407, -0.5156, -0.3710, -0.4400, -0.4654, -0.2721],
}
GEM = os.path.join(ROOT, "paper_reproduction", "s9_networked_systems", "gemini_weights_m5.json")
GEM_V2 = os.path.join(ROOT, "paper_reproduction", "s6_7_8_by_model", "10arg_gemini_a6v2", "gemini_a6v2_calib.json")

S = os.path.join(ROOT, "paper_reproduction", "s9_networked_systems", "data")
EXT = os.path.join(ROOT, "extension_replace_a6", "data")
BY = os.path.join(ROOT, "paper_reproduction", "s6_7_8_by_model")

CONFIG = {
    ("gpt", "a6"): dict(
        opinions=f"{ROOT}/paper_reproduction/s6_arguments_to_opinions/data/opinions_gpt.csv",
        sending=f"{ROOT}/paper_reproduction/s7_sending/data/sending_gpt.csv",
        receiving=f"{ROOT}/paper_reproduction/s8_receiving/data/receiving_gpt.csv",
        s9_llm=[f"{S}/n30_merged/s9_original_llm_shuffled_n30.pkl"],
        s9_classic=[f"{S}/n30_merged/s9_original_classic_shuffled_n30.pkl"],
        N=20, alpha=GPT_W["a6"], a6key="original", s9_note="reshuffle, matched, gpt-4.1, N=20"),
    ("gpt", "a6v2"): dict(
        opinions=f"{EXT}/repro_s6_a6v2_opinions_gpt.csv",
        sending=f"{EXT}/s7_sending.csv",
        receiving=f"{EXT}/s8_receiving_faithful.csv",
        s9_llm=[f"{EXT}/s9_a6v2_llm_shuffled_exp0_9.pkl", f"{EXT}/s9_a6v2_llm_shuffled_exp10_29.pkl"],
        s9_classic=[f"{EXT}/s9_a6v2_classic_shuffled_exp0_9.pkl", f"{EXT}/s9_a6v2_classic_shuffled_exp10_29.pkl"],
        N=20, alpha=GPT_W["a6v2"], a6key="v2_no_training", s9_note="reshuffle, matched, gpt-4.1, N=20"),
    ("gemini", "a6"): dict(
        opinions=f"{BY}/10arg_gemini/data/opinions_gemini.csv",
        sending=f"{BY}/10arg_gemini/data/sending_gemini.csv",
        receiving=f"{BY}/10arg_gemini/data/receiving_gemini.csv",
        s9_llm=[f"{S}/s9_n100_gemini_exp0.pkl", f"{S}/s9_n100_gemini_exp1_9.pkl", f"{S}/s9_n100_gemini_exp5_9.pkl",
                f"{S}/s9_n100_gemini_exp10_12.pkl", f"{S}/s9_n100_gemini_exp13_19.pkl"],
        s9_classic=[f"{S}/s9_n100_gemini_classic_exp0.pkl", f"{S}/s9_n100_gemini_classic_exp1_9.pkl",
                    f"{S}/s9_n100_gemini_classic_exp10_19.pkl"],
        N=100, alpha=json.load(open(GEM))["weights_m5"], a6key="original", s9_note="reshuffle, matched, gemini-flash-lite, N=100"),
    ("gemini", "a6v2"): dict(
        opinions=f"{BY}/10arg_gemini_a6v2/data/opinions_gemini.csv",
        sending=f"{BY}/10arg_gemini_a6v2/data/sending_gemini.csv",
        receiving=f"{BY}/10arg_gemini_a6v2/data/receiving_gemini.csv",
        s9_llm=[f"{S}/s9_n100_gemini_a6v2_llm_exp0_8.pkl", f"{S}/s9_n100_gemini_a6v2_llm_exp9_19.pkl"],
        s9_classic=[f"{S}/s9_n100_gemini_a6v2_classic_exp0_19.pkl"],
        N=100, alpha=json.load(open(GEM_V2))["weights_m5"], a6key="v2_no_training", s9_note="reshuffle, matched, gemini-flash-lite, N=100"),
}


def load_results(files):
    res = {}
    for f in files:
        for e, v in pickle.load(open(f, "rb"))["results"].items():
            res[e] = v
    return {e: v for e, v in res.items()
            if all(len(v[p][g]) == 3 for p in PROTOCOLS for g in (0, 1))}


def mean_ci(mat):
    """mat: runs x T (nan-padded). -> mean, 95% CI half-width (Student-t)."""
    mean = np.nanmean(mat, 0)
    n = np.sum(~np.isnan(mat), 0).astype(float)
    sd = np.nanstd(mat, 0, ddof=1)
    half = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(np.maximum(n, 1))
    return mean, half


def agg_s9(cfg):
    args = build_args(cfg["a6key"])
    idx = {a: i for i, a in enumerate(args)}
    alpha = np.array(cfg["alpha"])
    N, half = cfg["N"], cfg["N"] // 2
    llm = load_results(cfg["s9_llm"])
    cl = load_results(cfg["s9_classic"])
    exps = sorted(set(llm) & set(cl))
    out = {"n_runs": len(exps), "N": N, "note": cfg["s9_note"]}

    def opinion_traj(cell):
        # recompute per-agent opinion (order-free mean of alpha over held args) at each step, then group means
        hist = cell[2]
        g1 = np.array([[np.mean([alpha[idx[a]] for a in hist[t][i]]) if hist[t][i] else 0.0
                        for i in range(half)] for t in range(len(hist))]).mean(1)
        g2 = np.array([[np.mean([alpha[idx[a]] for a in hist[t][i]]) if hist[t][i] else 0.0
                        for i in range(half, N)] for t in range(len(hist))]).mean(1)
        return g1, g2

    for p in PROTOCOLS:
        for gt in (0, 1):
            for mode, data in (("llm", llm), ("cl", cl)):
                g1s, g2s = [], []
                for e in exps:
                    a, b = opinion_traj(data[e][p][gt])
                    g1s.append(a); g2s.append(b)
                L = max(len(x) for x in g1s)
                def pad(rows):
                    m = np.full((len(rows), L), np.nan)
                    for i, r in enumerate(rows):
                        m[i, :len(r)] = r
                    return m
                key = f"{p}|topo{gt}|{mode}"
                out[f"{key}|g1_mean"], out[f"{key}|g1_ci"] = mean_ci(pad(g1s))
                out[f"{key}|g2_mean"], out[f"{key}|g2_ci"] = mean_ci(pad(g2s))

    # argument-exchange distribution (window-averaged % of each arg among transmissions)
    W, n_args = 20 if N == 20 else 100, len(args)
    for p in PROTOCOLS:
        for gt in (0, 1):
            for mode, data in (("llm", llm), ("cl", cl)):
                per = []
                for e in exps:
                    comm = data[e][p][gt][0]
                    nw = (len(comm) + W - 1) // W
                    c = np.zeros((max(nw, 1), n_args))
                    for t, rec in enumerate(comm):
                        if rec[4] in idx:
                            c[t // W, idx[rec[4]]] += 1
                    s = c.sum(1, keepdims=True); s[s == 0] = 1
                    per.append(c / s * 100)
                T = max(len(x) for x in per)
                M = np.full((len(per), T, n_args), np.nan)
                for i, x in enumerate(per):
                    M[i, :len(x)] = x
                out[f"argdist|{p}|topo{gt}|{mode}"] = np.nanmean(M, 0)
    out["window"] = W

    # Fig 10 — cumulative Group-2 acceptance of each PRO argument, insert-to-the-beginning + Erdős-Rényi.
    # verdict 1 = newly accepted (2 = the receiver already held it, 0 = rejected); divided by the number
    # of Group-2 agents, accumulated over the same windows, averaged over runs.
    n_pro = n_args // 2                                   # PRO arguments, not agents
    curves = []
    for e in exps:
        comm = llm[e]["insert-to-the-beginning"][1][0]
        nw = (len(comm) + W - 1) // W
        c = np.zeros((max(nw, 1), n_pro))
        for t, rec in enumerate(comm):
            j = idx.get(rec[4])
            if rec[2] == 1 and rec[1] >= half and j is not None and j < n_pro:
                c[t // W, j] += 1
        curves.append(np.cumsum(c, axis=0) / half)
    T = min(len(x) for x in curves)
    out["fig10_curve"] = np.mean([x[:T] for x in curves], axis=0)
    return out


def main(only=None):
    """only: optional list of '<model>_<set>' keys to re-aggregate (default: all four)."""
    alpha_dump = json.load(open(os.path.join(DATA, "alpha.json"))) if (
        only and os.path.exists(os.path.join(DATA, "alpha.json"))) else {}
    for (model, s), cfg in CONFIG.items():
        if only and f"{model}_{s}" not in only:
            continue
        d = os.path.join(DATA, f"{model}_{s}")
        os.makedirs(d, exist_ok=True)
        for kind in ("opinions", "sending", "receiving"):
            src = cfg[kind]
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, f"{kind}.csv"))
            else:
                print(f"  !! missing {kind}: {src}")
        agg = agg_s9(cfg)
        curve = agg.pop("fig10_curve")
        np.savez_compressed(os.path.join(d, "s9_agg.npz"), **agg)
        np.savez_compressed(os.path.join(d, "fig10.npz"), curve=curve,
                            window=agg["window"], n_runs=agg["n_runs"])
        alpha_dump[f"{model}_{s}"] = {"alpha_m5": [round(float(x), 5) for x in cfg["alpha"]],
                                      "a6_variant": cfg["a6key"], "s9_N": cfg["N"], "s9_n_runs": int(agg["n_runs"])}
        print(f"OK {model}_{s}: CSVs + s9_agg.npz (n_runs={agg['n_runs']}, N={cfg['N']})")
    json.dump(alpha_dump, open(os.path.join(DATA, "alpha.json"), "w"), indent=2)
    # arg-set texts (a6 string per set) for the README / reference
    texts = {"a6_original": A6_VARIANTS["original"], "a6_v2_no_training": A6_VARIANTS["v2_no_training"]}
    json.dump(texts, open(os.path.join(DATA, "a6_variant_texts.json"), "w"), indent=2)
    print("wrote data/alpha.json + data/a6_variant_texts.json")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)

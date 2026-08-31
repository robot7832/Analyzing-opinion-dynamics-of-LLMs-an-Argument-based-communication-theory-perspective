# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
AnglE-only argument-pair similarity figures (NO GTE — single-encoder signal).

Uses AnglE/UAE-Large-V1, calibrated to the human STS-B scale (isotonic cosine -> 0-5), to plot the
pairwise calibrated similarity of the arguments. Two figures:

  fig_anglE_pairs_by_valence.png  — the 45 pairs per generation experiment, coloured by pair valence
                                    (pro-pro red, con-con blue, pro-con purple); red diamond = mean.
  fig_anglE_a6_neighbours.png     — the paper set's 45 pairs (grey), with a6-a7 / a6-a8 / a6-a9 in red.

Run in the pinned encoder venv (transformers 4.46; see ../requirements-encoders.txt):
    ~/.venvs/embed_sts/bin/python anglE_pair_plots.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
CACHE = HERE / "data" / "anglE_cache.npz"
DEVICE = "cpu"
HF_ID = "WhereIsAI/UAE-Large-V1"          # AnglE/UAE — cls pooling, no remote code

# ── argument sets (the four generation experiments; pro = first 5, con = last 5 for valence sets) ──
ARGS1 = [  # paper set a1..a10 (with "independent") — the headline set
    "Individuals have a fundamental right to protect themselves, their families, and their property from immediate threats when law enforcement cannot arrive in time.",
    "The Second Amendment explicitly protects this right as part of the foundational legal framework, and constitutional rights should be preserved regardless of contemporary policy preferences.",
    "An armed citizenry serves as a check against potential government overreach and protects democratic institutions from authoritarian drift.",
    "People in remote areas often require firearms for protection against wildlife, pest control, and situations where emergency services are hours away.",
    "Firearms allow physically weaker individuals to defend themselves against stronger attackers, providing equal protection regardless of physical capabilities.",
    "Countries with stricter gun laws consistently show lower rates of gun violence, homicides, and mass shootings, demonstrating that restrictions save lives.",
    "Widespread gun ownership leads to preventable deaths through accidents, particularly involving children who access improperly stored weapons.",
    "Easy access to firearms significantly increases successful suicide attempts, as guns provide a highly lethal means during moments of crisis.",
    "Firearms in homes increase the likelihood that domestic disputes will result in fatalities, particularly endangering women in abusive relationships.",
    "The original intent of armed militias is obsolete given professional military and police forces, making civilian gun ownership an outdated concept that no longer serves its historical purpose.",
]
ARGS2 = [  # without the word "independent"
    "Individuals have a fundamental right to protect themselves, their families, and their property from criminals who may be armed regardless of laws.",
    "In countries like the US, this right is explicitly protected by founding documents (Second Amendment) as an essential liberty that should not be infringed.",
    "An armed citizenry serves as a check against potential government overreach and protects democratic institutions from authoritarian threats.",
    "People in rural areas often rely on firearms for protection from wildlife, pest control, and situations where police response times may be very long.",
    "Firearms play important roles in hunting traditions, competitive shooting sports, and cultural practices that have been passed down through generations.",
    "Widespread gun ownership increases risks of accidents, impulsive violence, domestic abuse fatalities, and mass shooting incidents that harm innocent people.",
    "Limiting access to firearms reduces suicide rates, as guns provide a highly lethal means during moments of crisis that might otherwise pass.",
    "Widespread civilian gun ownership makes police work more dangerous and can lead to escalation of routine encounters into deadly situations.",
    "Gun rights were conceived in a different era; modern weapons are far more deadly than those envisioned by historical lawmakers.",
    "Countries with stricter gun laws often have significantly lower rates of gun violence while maintaining personal freedoms and democratic governance.",
]
ARGS3 = [  # 5 synonym pairs (no pro/con structure)
    "The right to keep and bear arms is fundamental to protecting individual liberty and preventing government tyranny, as an armed citizenry serves as the ultimate check against authoritarian overreach.",
    "Gun ownership rights are essential for safeguarding personal freedom and deterring oppressive government control, since citizens with firearms provide the final barrier against dictatorial abuse of power.",
    "Individuals have an inherent right to defend themselves, their families, and their property from criminals and other threats, which requires access to effective means of protection including firearms.",
    "People possess a natural right to protect themselves, their loved ones, and their belongings from violent criminals and other dangers, necessitating the ability to obtain and use guns for self-defense.",
    "The Second Amendment to the Constitution explicitly guarantees Americans the right to keep and bear arms, making gun ownership a legally protected constitutional right that should not be infringed.",
    "Gun ownership is constitutionally protected under the Second Amendment, which clearly establishes that Americans have a fundamental legal right to possess and carry firearms that must be respected.",
    "Responsible gun ownership by law-abiding citizens actually enhances public safety by deterring crime, as criminals are less likely to target victims who may be armed and able to defend themselves.",
    "When lawful citizens carry firearms responsibly, it improves community security by discouraging criminal activity, since perpetrators avoid potential victims who might be equipped to fight back.",
    "Firearms serve important practical purposes for many Americans, including hunting for food and sport, pest control on farms and ranches, and recreational target shooting activities.",
    "Guns have legitimate utilitarian functions in American society, such as hunting animals for sustenance and recreation, managing wildlife on agricultural properties, and engaging in competitive marksmanship sports.",
]
ARGS4 = [  # 5 pro-synonyms + 5 con-synonyms (two stance clusters)
    "Citizens must retain the ability to defend themselves against potential government tyranny and oppression.",
    "The populace needs to maintain the means to resist authoritarian overreach by their own government.",
    "People should preserve their capacity to protect themselves from potential state-sponsored persecution.",
    "Individuals require the power to safeguard their freedoms against possible governmental abuse of authority.",
    "The public must possess the tools necessary to counter any future despotic actions by their rulers.",
    "Widespread civilian access to firearms significantly increases the likelihood of deadly violence in communities.",
    "When ordinary people can easily obtain guns, it substantially raises the probability of lethal harm occurring in society.",
    "Broad public availability of weapons creates a much higher risk of fatal incidents among the general population.",
    "Easy civilian access to firearms dramatically elevates the chances of people being killed or seriously injured.",
    "When guns are readily accessible to citizens, it greatly amplifies the potential for tragic loss of life in everyday situations.",
]
EXP = {
    "1_paper(indep)":      dict(args=ARGS1, kind="valence"),
    "2_without_indep":     dict(args=ARGS2, kind="valence"),
    "3_synonym_pairs":     dict(args=ARGS3, kind="pairs"),
    "4_synonym_by_stance": dict(args=ARGS4, kind="valence"),
}
LABELS = [f"a{i+1}" for i in range(10)]


@torch.no_grad()
def encode(model, tok, texts, bs=64, max_length=512):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(DEVICE)
        cls = model(**enc).last_hidden_state[:, 0]                       # cls pooling
        out.append(torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy())
    return np.vstack(out)


def build_calibrator_and_embeddings():
    """Return (isotonic calibrator, {exp: arg embeddings}, STS-B Spearman) — cached after first run."""
    tok = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModel.from_pretrained(HF_ID).to(DEVICE).eval()
    # STS-B calibration (the slow part) — cache the raw cosines + gold
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        sb_cos, sb_gold = z["sb_cos"], z["sb_gold"]
        emb = {e: z[f"emb_{e}"] for e in EXP}
    else:
        from datasets import load_dataset
        stsb = load_dataset("sentence-transformers/stsb", split="test")
        s1, s2 = list(stsb["sentence1"]), list(stsb["sentence2"])
        SBE = encode(model, tok, s1 + s2, max_length=128); n = len(s1)
        sb_cos = (SBE[:n] * SBE[n:]).sum(1)
        sb_gold = np.asarray(stsb["score"], dtype=float) * 5.0
        emb = {e: encode(model, tok, EXP[e]["args"]) for e in EXP}
        np.savez(CACHE, sb_cos=sb_cos, sb_gold=sb_gold, **{f"emb_{e}": emb[e] for e in EXP})
    iso = IsotonicRegression(out_of_bounds="clip").fit(sb_cos, sb_gold)
    rho = spearmanr(sb_cos, sb_gold).correlation * 100
    return iso, emb, rho


def calibrated_matrix(iso, E):
    cos = np.clip(E @ E.T, -1, 1)
    return np.clip(iso.predict(cos.ravel()).reshape(cos.shape), 0, 5)


def upper_pairs(M):
    iu = np.triu_indices(M.shape[0], 1)
    return list(zip(iu[0], iu[1], M[iu]))


# colours
C_PP, C_CC, C_PC = "#c0392b", "#2c6fbb", "#8e44ad"      # pro-pro, con-con, pro-con
C_GREY, C_RED = "#9aa3ab", "#c0392b"


def pair_color(i, j):
    if i < 5 and j < 5:
        return C_PP
    if i >= 5 and j >= 5:
        return C_CC
    return C_PC


# one fixed within-column jitter per experiment, so the cloud is identical across the variants
JITTER = {e: np.random.default_rng(k).uniform(-0.16, 0.16, 45) for k, e in enumerate(EXP)}


def fig_by_valence(iso, emb, rho, exps, colored: bool, fname: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for lo, hi, c in [(0, 1, "#e9f3e9"), (1, 2, "#fbf3df"), (2, 3, "#fce6d6"), (3, 4, "#f7d9d9"), (4, 5, "#f0c4c4")]:
        ax.axhspan(lo, hi, color=c, zorder=0)
    for xi, e in enumerate(exps):
        pairs = upper_pairs(calibrated_matrix(iso, emb[e]))
        xs = xi + JITTER[e][:len(pairs)]
        colors = [pair_color(i, j) for i, j, _ in pairs] if colored else C_GREY
        ax.scatter(xs, [v for _, _, v in pairs], s=26, alpha=0.75, zorder=3, c=colors, edgecolors="none")
        ax.scatter([xi], [np.mean([v for _, _, v in pairs])], s=200, marker="D",
                   color="black", edgecolors="white", linewidths=1.2, zorder=4)
    ax.set_xticks(range(len(exps))); ax.set_xticklabels(list(exps), rotation=12)
    ax.set_ylim(0, 5); ax.set_ylabel("calibrated STS (0-5)")
    if colored:
        handles = [plt.Line2D([], [], marker="o", ls="", color=C_PP, label="pro-pro"),
                   plt.Line2D([], [], marker="o", ls="", color=C_CC, label="con-con"),
                   plt.Line2D([], [], marker="o", ls="", color=C_PC, label="pro-con"),
                   plt.Line2D([], [], marker="D", ls="", color="black", mec="white", label="mean")]
    else:
        handles = [plt.Line2D([], [], marker="o", ls="", color=C_GREY, label="argument pair"),
                   plt.Line2D([], [], marker="D", ls="", color="black", mec="white", label="mean")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"AnglE-only calibrated similarity of the 45 argument pairs per experiment\n"
                 f"({subtitle}; STS-B Spearman {rho:.1f})", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)
    print(f"saved {fname}")


def fixed_pair_x(M):
    """One fixed x-jitter per pair (i, j), identical across all neighbour figures."""
    rng = np.random.default_rng(0)
    return {(i, j): float(rng.uniform(-0.32, 0.32)) for i, j, _ in upper_pairs(M)}


def fig_center_neighbours(M, pair_x, center: int, partners: list[int]):
    """Paper-set 45 pairs (fixed positions); only the colouring changes per figure.

    The grey cloud is identical in every figure — pair_x fixes each point's x — and the three
    (center, partner) pairs are recoloured red in place.
    """
    cname = LABELS[center]
    highlight = {tuple(sorted((center, p))): f"{cname}-{LABELS[p]}" for p in partners}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for lo, hi, c in [(0, 1, "#e9f3e9"), (1, 2, "#fbf3df"), (2, 3, "#fce6d6"), (3, 4, "#f7d9d9"), (4, 5, "#f0c4c4")]:
        ax.axhspan(lo, hi, color=c, zorder=0)
    for i, j, v in upper_pairs(M):                      # every point stays put; colour is the only change
        hl = (i, j) in highlight
        ax.scatter([pair_x[(i, j)]], [v], s=80 if hl else 34, zorder=5 if hl else 3,
                   color=C_RED if hl else C_GREY, edgecolors="black" if hl else "none",
                   linewidths=0.9 if hl else 0, alpha=1.0 if hl else 0.7)
    # labels on the right with leader lines, de-collided vertically (points are NOT moved)
    hl_sorted = sorted(highlight.items(), key=lambda kv: M[kv[0][0], kv[0][1]])
    prev_y = -9.0
    for (i, j), name in hl_sorted:
        v = M[i, j]
        ly = max(v, prev_y + 0.34); prev_y = ly
        ax.annotate(f"{name} = {v:.2f}", xy=(pair_x[(i, j)], v), xytext=(0.46, ly),
                    fontsize=9, color=C_RED, weight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=C_RED, lw=0.8))
    ax.axhline(M[~np.eye(10, dtype=bool)].mean(), ls="--", color="black", lw=1, zorder=2)
    ax.set_xlim(-0.55, 1.05); ax.set_xticks([]); ax.set_ylim(0, 5)
    ax.set_ylabel("calibrated STS (0-5)")
    ax.set_title(f"AnglE-only: a1-a10 pairwise similarity\n(grey = 45 pairs; red = {cname}'s harm-cluster "
                 f"pairs; dashed = mean)")
    fig.tight_layout(); fig.savefig(FIG / f"fig_anglE_{cname}_neighbours.png", dpi=130); plt.close(fig)
    print(f"saved fig_anglE_{cname}_neighbours.png  ({', '.join(f'{n}={M[i,j]:.2f}' for (i,j),n in highlight.items())})")


def main():
    iso, emb, rho = build_calibrator_and_embeddings()
    print(f"AnglE/UAE loaded; STS-B Spearman = {rho:.1f}")
    all_exps = list(EXP)
    no_synpairs = [e for e in EXP if e != "3_synonym_pairs"]
    fig_by_valence(iso, emb, rho, all_exps, True,  "fig_anglE_pairs_by_valence.png", "coloured by valence")
    fig_by_valence(iso, emb, rho, all_exps, False, "fig_anglE_pairs_all_grey.png", "all grey")
    fig_by_valence(iso, emb, rho, no_synpairs, True, "fig_anglE_pairs_by_valence_no_synpairs.png",
                   "coloured by valence, synonym_pairs excluded")
    M = calibrated_matrix(iso, emb["1_paper(indep)"])   # paper-set matrix, used by all neighbour figs
    pair_x = fixed_pair_x(M)                             # one fixed layout shared across the figures
    cluster = [5, 6, 7, 8]                               # a6, a7, a8, a9 (the gun-harm cluster)
    for center in cluster:
        fig_center_neighbours(M, pair_x, center, [p for p in cluster if p != center])


if __name__ == "__main__":
    main()

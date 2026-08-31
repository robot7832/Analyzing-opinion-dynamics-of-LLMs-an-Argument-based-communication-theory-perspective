# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§9 opinion dynamics — mean ± 95 % CI (self-contained; only numpy + matplotlib).

`s9_agg.npz` stores, per (protocol × topology × mode × group), the mean trajectory and the
**Student-t 95 % CI half-width** of that mean (`t.ppf(0.975, n-1) · sd / sqrt(n)`), so the band is
drawn directly from the `*_ci` arrays — no conversion needed.

    python fig_s9_ci.py            # renders gpt_a6v2 and gpt_a6
"""
import os

import matplotlib.pyplot as plt
import numpy as np

PROTOCOLS = ["insert-to-end", "insert-to-the-beginning"]
PROTO_LABEL = {"insert-to-end": "insert-to-the-end",
               "insert-to-the-beginning": "insert-to-the-beginning"}
TOPO = {0: "Topology 1 (dense clusters)", 1: "Topology 2 (Erdős–Rényi)"}


def _load_agg(data_dir):
    z = np.load(os.path.join(data_dir, "s9_agg.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def fig_s9(data_dir, model, out, FS_value=13, LS_value=11):
    """Four panels (protocol × topology); LLM vs matched classical, mean ± 95 % CI."""
    z = _load_agg(data_dir)
    n_runs, N9 = int(z["n_runs"]), int(z["N"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    panel_label = ["a", "b", "c", "d"]

    for row, p in enumerate(PROTOCOLS):
        for col, gt in enumerate((0, 1)):
            ax = axes[row, col]
            for mode, c1, c2, lab in (("llm", "#d62728", "#1f77b4", "LLM"),
                                      ("cl", "#ff7f0e", "#9467bd", "classical")):
                key = f"{p}|topo{gt}|{mode}"
                for grp, col_ in (("g1", c1), ("g2", c2)):
                    m, h = z[f"{key}|{grp}_mean"], z[f"{key}|{grp}_ci"]   # h = 95 % CI half-width
                    t = np.arange(len(m))
                    gname = "Group 1" if grp == "g1" else "Group 2"
                    ax.plot(t, m, "-", color=col_, lw=2.2 if mode == "llm" else 1.4,
                            label=f"{lab} {gname}", zorder=4)
                    ax.fill_between(t, m - h, m + h, color=col_,
                                    alpha=0.25 if mode == "llm" else 0.13, lw=0)

            ax.axhline(0, color="#888", lw=0.7, ls="--", alpha=0.6)
            ax.set_title(f"({panel_label[2 * row + col]}) {TOPO[gt]}\n{PROTO_LABEL[p]}",
                         fontsize=FS_value)
            ax.grid(True, alpha=0.25)

            # force y tick labels on every panel despite sharey=True,
            # and bring tick label size in line with FS_value
            ax.tick_params(axis="both", labelsize=FS_value - 2, labelleft=True)

            if row == 1:
                ax.set_xlabel("time", fontsize=FS_value)
            if col == 0:
                ax.set_ylabel("opinion", fontsize=FS_value)
            if (row, col) == (0, 0):
                ax.legend(fontsize=LS_value, framealpha=0.9)

    fig.suptitle(f"{model} — N = {N9}, {n_runs} runs, mean ± 95 % CI",
                 fontsize=FS_value + 2, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"n_runs": n_runs, "N": N9}


if __name__ == "__main__":
    COLL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "paper_reproduction", "a6_vs_a6v2_figures", "data")
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "deck")
    for tag, name in (("gpt_a6v2", "gpt-4.1 · a6v2"), ("gpt_a6", "gpt-4.1 · a6")):
        print(tag, fig_s9(os.path.join(COLL, tag), name,
                          os.path.join(OUT, f"s9_fig7_ci_{tag}.png")))

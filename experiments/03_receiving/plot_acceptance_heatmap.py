# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
The confirmation-bias map — one 2x2 figure with the four §8 acceptance-rate heatmaps:

    (a) gpt-4.1 (gun control)      (b) gemini-2.5-flash-lite (gun control)
    (c) gpt-4.1 (calorie count)    (d) claude-sonnet-4       (calorie count)

Every panel is the paper's 10-argument scaffold (5 pro a1–a5 / 5 con a6–a10) so the four are
directly comparable cell for cell, and every panel is averaged over **all three** sender settings
(human / llm / blinded) — the "(all)" of the layout sketch.

Styling follows the maintainer's `fig_s8_heatmap`: rocket colormap on a fixed 0–1 scale, `_fmt`
annotations ('0', '1', '.57'), white gridlines, and the blue pro/con boundary at the 5-argument
split. The per-panel colorbars of the single-panel version are replaced by one shared bar, since
all four share the 0–1 scale.

    /opt/anaconda3/bin/python fig_bias_map.py     -> figures/bias_map.{png,pdf} + bias_map.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from extra_figs import load_receiving

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
COLL = ROOT / "paper_reproduction" / "a6_vs_a6v2_figures" / "data"
FIG = HERE / "figures"; FIG.mkdir(parents=True, exist_ok=True)

N = 10
HALF = N // 2

# (panel letter, title, receiving.csv folder) — the layout of the sketch, row-major.
PANELS = [
    ("a", "gpt-4.1 (gun control)", COLL / "gpt_a6"),
    ("b", "gemini-2.5-flash-lite (gun control)", COLL / "gemini_a6"),
    ("c", "gpt-4.1 (calorie count)", HERE / "data" / "gpt_calories"),
    ("d", "claude-sonnet-4 (calorie count)", HERE / "data" / "claude_calories"),
]


def _fmt(v):
    """0.00 -> '0', 1.00 -> '1', 0.57 -> '.57'"""
    if np.isnan(v):
        return ""
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}".lstrip("0")


def fig_bias_map(out, FS_value=13, TS_value=11):
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 13.6))
    stats, mesh = {}, None

    for ax, (letter, title, d) in zip(axes.ravel(), PANELS):
        df = load_receiving(os.path.join(d, "receiving.csv"), N)
        mat = (df.groupby(["held_i", "received_i"]).endorse.mean().unstack()
               .reindex(index=range(N), columns=range(N)))

        sns.heatmap(mat, vmin=0, vmax=1,
                    annot=np.vectorize(_fmt)(mat.values), fmt="",
                    cmap="rocket", square=True,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"fontsize": TS_value - 1}, ax=ax, cbar=False)
        mesh = ax.collections[0]

        ax.set_xticks(np.arange(N) + 0.5)
        ax.set_yticks(np.arange(N) + 0.5)
        ax.set_xticklabels([f"$a_{{{k}}}$" for k in range(1, N + 1)],
                           fontsize=TS_value, rotation=0)
        ax.set_yticklabels([f"$a_{{{k}}}$" for k in range(1, N + 1)],
                           fontsize=TS_value, rotation=0)
        ax.axhline(HALF, color="#3d8bfd", lw=2.0)          # pro/con boundary
        ax.axvline(HALF, color="#3d8bfd", lw=2.0)
        ax.set_xlabel("sending argument", fontsize=FS_value)   # Figure 5's wording
        ax.set_ylabel("receiver's argument", fontsize=FS_value)
        ax.set_title(f"({letter})  {title}", fontsize=FS_value + 1, pad=9)

        same = df[df.held_val == df.received_val].endorse.mean()
        opp = df[df.held_val != df.received_val].endorse.mean()
        stats[letter] = {"panel": title,
                         "same_accept": round(float(same), 3),
                         "opp_accept": round(float(opp), 3)}

    fig.tight_layout(rect=(0, 0, 0.935, 1))
    cax = fig.add_axes([0.945, 0.16, 0.016, 0.68])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("acceptance rate", fontsize=FS_value)
    cb.ax.tick_params(labelsize=TS_value)

    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


def main():
    stats = fig_bias_map(FIG / "bias_map.png")
    json.dump(stats, open(HERE / "bias_map.json", "w"), indent=2)
    for k, v in stats.items():
        print(f"  ({k}) {v['panel']:38s} same {v['same_accept']:.3f}   opposite {v['opp_accept']:.3f}")
    print(f"\nsaved {FIG / 'bias_map.png'} (+ .pdf) and bias_map.json")


if __name__ == "__main__":
    main()

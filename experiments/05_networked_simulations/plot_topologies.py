# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§9 — the two interaction networks, with every agent coloured by its initial opinion.

    (a) Topology 1 (dense clusters)   — SBM, in-group edge p = 0.5, between-group p = 0.1 (footnote 7)
    (b) Topology 2 (Erdős–Rényi)      — the same edge probability everywhere

Companion to the manuscript's Figure 6: same data (the matched n = 30 gpt-4.1 collection), same
wording for the panel titles, and the same Group-1-red / Group-2-blue that Figure 6 uses for the
LLM trajectories, here interpolated through white to make the colour scale.

Both graphs are the pair experiment `EXP` actually ran — `simulate_network.make_graphs(
random.Random(1000 + exp))`, redrawn until both are connected — so they are reproduced, not
invented. Node colour is the agent's opinion at t = 0, read off the t = 0 row of the matched run's
argument history and scored with the model-(5) α of Figure 2b (the same α that produces the
trajectories in Figure 6). The LLM and classical arms share the seed, so that row is identical in
both, which is what "the same initial opinion endowment" in the Figure 6 caption refers to.

`EXP` is not arbitrary, and it is chosen on two criteria in this order:

1. **Equal edge counts.** Sec. 9 defines Topology 2 as an Erdős–Rényi graph with the same number of
   edges as Topology 1, but `make_graphs` draws it at p = |E_SBM| / C(N,2), which matches that count
   only *in expectation* — across the 30 experiments the two graphs differ by up to 10 edges. Only
   experiments 29 and 30 hit equality exactly (47 and 45 edges), so a figure that puts the two
   graphs side by side has to come from one of those two; anything else invites the reader to read
   an accidental density difference as part of the design.
2. **Representative endowment, with every agent endowed.** Of the draws that survive (1), this one
   sits closest to the initialisation the manuscript reports in para. 9.12 — Group 1 ≈ +0.47,
   Group 2 ≈ −0.24 — among those in which no agent starts with an empty argument set. Empty
   agents do occur (para. 9.9) and are not an error, but they are not marked here, and an unmarked
   empty agent would be coloured white and read as "opinion 0" rather than "holds nothing". Both
panels are coloured by that single draw (`SHARED_INIT`), so the only thing differing between them
is the wiring. Set `SHARED_INIT = False` to give each panel its own condition's real t = 0 instead:
`run_condition`'s seed is `10000·exp + 100·(protocol == "insert-to-end") + topology`, so every cell
of the 2 × 2 draws its own endowment and the two panels then differ in colour as well.

    /opt/anaconda3/bin/python fig_s9_topologies.py
        -> figures/s9_network_topologies.{png,pdf} + s9_network_topologies.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
S9 = ROOT / "paper_reproduction" / "s9_networked_systems"
FIG = HERE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(S9))
sys.path.insert(0, str(ROOT / "src"))

import simulate_network as sn  # noqa: E402  (make_graphs / connected — the graphs are not re-derived)
from act.arguments import build_args  # noqa: E402

# --- the collection behind Figure 6 (gpt-4.1 · A_gc, N = 20, matched, reshuffled, n = 30) ---
D30 = S9 / "data" / "n30_merged"
LLM_PKL = D30 / "s9_original_llm_shuffled_n30.pkl"
CL_PKL = D30 / "s9_original_classic_shuffled_n30.pkl"
ALPHA = sn.WEIGHTS["original"]          # model-(5) α, Figure 2b
EXP, PROTO = 30, "insert-to-the-beginning"   # |E| = 45 in both topologies; see the module docstring
INIT_TOPO = 1                           # which of that experiment's t = 0 draws colours both panels
SHARED_INIT = True                      # False: colour each panel by its own condition's t = 0
PAPER_INIT = (0.47, -0.24)              # group means stated in para. 9.12

N9 = 20
HALF = N9 // 2
FS_value, LS_value = 13, 11             # Figure 6's typography
TOPO = {0: "Topology 1 (dense clusters)", 1: "Topology 2 (Erdős–Rényi)"}
PROTO_LABEL = {"insert-to-end": "insert-to-the-end",
               "insert-to-the-beginning": "insert-to-the-beginning"}
C_G1, C_G2 = "#d62728", "#1f77b4"       # Figure 6's LLM Group 1 / Group 2
CMAP = LinearSegmentedColormap.from_list("g2_white_g1", [C_G2, "#ffffff", C_G1])
VMAX = 0.9                              # max |α| = 0.897 (a6), so ±0.9 spans the reachable range
E_IN, E_OUT = "#c8c8c8", "#6f6f6f"      # within-group / between-group edge


def node_positions():
    """Two blobs of 10, Group 1 left and Group 2 right; identical in both panels, so a node keeps
    its place and only the wiring changes between the topologies."""
    def blob(cx, r=1.05, n=HALF):
        ang = np.deg2rad(90 - np.arange(n) * 360 / n)
        return np.c_[cx + r * np.cos(ang), r * np.sin(ang)]
    return np.vstack([blob(-1.55), blob(+1.55)])


def graphs_of(exp):
    """The (SBM, ER) pair of experiment `exp`, exactly as simulate_network builds it."""
    grng = random.Random(1000 + exp)
    while True:
        g = sn.make_graphs(grng)
        if sn.connected(g[0]) and sn.connected(g[1]):
            return g


def initial_state(exp, protocol):
    """{topology: (opinions, |S|)} at t = 0, from the matched run's own argument history."""
    llm = pickle.load(open(LLM_PKL, "rb"))["results"]
    cl = pickle.load(open(CL_PKL, "rb"))["results"]
    idx = {a: i for i, a in enumerate(build_args("original"))}
    out = {}
    for gt in (0, 1):
        rows = []
        for res in (llm, cl):
            held = res[exp][protocol][gt][2][0]
            rows.append([float(np.mean([ALPHA[idx[a]] for a in s])) if s else 0.0 for s in held])
        assert np.allclose(rows[0], rows[1]), "the matched pair must share the initial endowment"
        out[gt] = (np.array(rows[0]), np.array([len(s) for s in llm[exp][protocol][gt][2][0]]))
    return out


def draw_panel(ax, edges, opinions, letter, gt, pos, norm):
    within = [(u, v) for u, v in edges if (u < HALF) == (v < HALF)]
    between = [(u, v) for u, v in edges if (u < HALF) != (v < HALF)]
    for group, col, lw in ((within, E_IN, 1.0), (between, E_OUT, 1.4)):
        for u, v in group:
            ax.plot(*zip(pos[u], pos[v]), color=col, lw=lw, zorder=1, solid_capstyle="round")

    for lo, hi, marker in ((0, HALF, "o"), (HALF, N9, "s")):
        sl = slice(lo, hi)
        ax.scatter(pos[sl, 0], pos[sl, 1], s=700, marker=marker, c=opinions[sl], cmap=CMAP,
                   norm=norm, edgecolors="#333333", linewidths=1.0, zorder=3)
    for i, (x, y) in enumerate(pos):
        shade = "white" if abs(opinions[i]) > 0.62 * VMAX else "#1a1a1a"
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=FS_value - 5,
                color=shade, zorder=4)

    for cx, lab, col in ((-1.55, "Group 1", C_G1), (1.55, "Group 2", C_G2)):
        ax.text(cx, 1.60, lab, ha="center", va="bottom", fontsize=FS_value, color=col)

    ax.set_title(f"({letter}) {TOPO[gt]}\n{len(edges)} edges, {len(between)} between groups",
                 fontsize=FS_value)
    ax.set_xlim(-2.85, 2.85); ax.set_ylim(-1.35, 2.0)
    ax.set_aspect("equal"); ax.axis("off")
    return len(within), len(between)


def main(exp=EXP, protocol=PROTO):
    graphs = graphs_of(exp)
    init = initial_state(exp, protocol)
    norm = Normalize(-VMAX, VMAX)

    # With the colour bar moved under the panels the axes span the full width, so the figure is
    # proportioned to keep the aspect="equal" box tight against the blobs in both directions.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    stats = {"exp": exp, "protocol": PROTO_LABEL[protocol], "N": N9, "n_runs_in_collection": 30,
             "alpha": "model-(5), Figure 2b", "shared_init": SHARED_INIT,
             "paper_para_9_12_init": list(PAPER_INIT), "vmax": VMAX, "topologies": {}}
    for ax, letter, gt in zip(axes, "ab", (0, 1)):
        op, sz = init[INIT_TOPO if SHARED_INIT else gt]
        n_in, n_out = draw_panel(ax, graphs[gt], op, letter, gt, node_positions(), norm)
        stats["topologies"][TOPO[gt]] = {
            "edges": n_in + n_out, "within_group": n_in, "between_group": n_out,
            "density": round((n_in + n_out) / (N9 * (N9 - 1) / 2), 3),
            "g1_mean_opinion_t0": round(float(op[:HALF].mean()), 3),
            "g2_mean_opinion_t0": round(float(op[HALF:].mean()), 3),
            "opinion_range_t0": [round(float(op.min()), 3), round(float(op.max()), 3)],
            "empty_agents_t0": [int(i + 1) for i in np.flatnonzero(sz == 0)],
        }

    handles = [Line2D([], [], marker="o", ls="none", mfc="#f2f2f2", mec="#333333",
                      ms=12, label="Group 1 agent"),
               Line2D([], [], marker="s", ls="none", mfc="#f2f2f2", mec="#333333",
                      ms=11, label="Group 2 agent"),
               Line2D([], [], color=E_IN, lw=2.2, label="within-group edge"),
               Line2D([], [], color=E_OUT, lw=2.2, label="between-group edge")]
    fig.legend(handles=handles, fontsize=LS_value, ncol=len(handles), frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.145))

    # The colour bar is left unlabelled and there is no caption block: what the colour and the run
    # are goes in the manuscript's figure caption, not into the artwork. The numbers that used to sit
    # in the caption (group means, edge counts, empty agents) are all in the printout and the JSON.
    fig.tight_layout(rect=(0, 0.21, 1, 1))
    cax = fig.add_axes([0.395, 0.075, 0.21, 0.030])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([-VMAX, -0.45, 0, 0.45, VMAX])
    cb.ax.tick_params(labelsize=FS_value - 2)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"s9_network_topologies.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)

    json.dump(stats, open(HERE / "s9_network_topologies.json", "w"), indent=2)
    print(f"experiment {exp}, {PROTO_LABEL[protocol]}, shared_init={SHARED_INIT}\n")
    for name, v in stats["topologies"].items():
        print(f"  {name:32s} {v['edges']:3d} edges ({v['between_group']:2d} between, "
              f"density {v['density']:.3f})   t=0  G1 {v['g1_mean_opinion_t0']:+.3f} / "
              f"G2 {v['g2_mean_opinion_t0']:+.3f}   empty {v['empty_agents_t0']}")
    n_edges = {v["edges"] for v in stats["topologies"].values()}
    print(f"\n  |E| equal in both topologies: {'yes' if len(n_edges) == 1 else '!! NO — ' + str(sorted(n_edges))}"
          "   (Topology 2 is defined as Erdős–Rényi with the same edge count)")
    print(f"  para. 9.12 initialisation: G1 {PAPER_INIT[0]:+.2f} / G2 {PAPER_INIT[1]:+.2f}")
    print("saved figures/s9_network_topologies.png (+ .pdf), s9_network_topologies.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, default=EXP)
    ap.add_argument("--protocol", default=PROTO, choices=list(PROTO_LABEL))
    a = ap.parse_args()
    main(a.exp, a.protocol)

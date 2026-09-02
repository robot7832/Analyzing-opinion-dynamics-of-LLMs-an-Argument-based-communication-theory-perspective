"""
Expand a packed `runs.npz` back into full opinion dynamics.

`runs.npz` stores each run as its ordered initial argument sets plus its communication log. The
per-step snapshots of every agent's set are not stored because they are a deterministic function of
those two: replaying the log under the run's insertion protocol reproduces them exactly, ordering
included. That keeps the whole simulation record at a few megabytes instead of a couple of gigabytes,
with nothing lost.

    python reconstruct.py --space A_gc --model gpt-4.1
    python reconstruct.py --space A_gc --model gpt-4.1 --protocol insert-to-end --topology 1 --run 0

Returned by `histories()` / `opinions()`:

    histories  list over runs of list over steps of list over agents of argument indices, in order
    opinions   array (n_runs, n_steps, n_agents) of the ACT opinion of every agent at every step

The opinion of an agent is the mean of the model-(5) weights of the arguments it holds, so it needs
the weights of the model whose run this is; pass them with --alpha, or read them from
`01_numerical_opinion`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
VERDICT_REJECTED, VERDICT_ACCEPTED, VERDICT_ALREADY_HELD = 0, 1, 2


def load(space: str, model: str):
    f = ROOT / "data" / "05_networked_simulations" / space / model / "runs.npz"
    if not f.exists():
        raise SystemExit(f"no packed runs at {f.relative_to(ROOT)}")
    return np.load(f, allow_pickle=True)


def cells(z) -> list[tuple[str, int, str]]:
    """Every (protocol, topology, arm) the file carries."""
    out = []
    for k in z.files:
        if k.startswith("log|"):
            _, protocol, topo, arm = k.split("|")
            out.append((protocol, int(topo[-1]), arm))
    return sorted(set(out))


def histories(z, protocol: str, topology: int, arm: str = "llm") -> list[list[list[list[int]]]]:
    """Replay the log; returns the ordered argument sets of every agent at every step."""
    init = z[f"init|{protocol}|topo{topology}|{arm}"]
    log = z[f"log|{protocol}|topo{topology}|{arm}"]
    to_end = protocol == "insert-to-end"
    runs = []
    for r in range(len(log)):
        sets = [[int(a) for a in row if a >= 0] for row in init[r]]
        snapshots = [[list(s) for s in sets]]
        for sender, receiver, verdict, argument in log[r]:
            if sender < 0:                                  # padding past the end of a short run
                break
            receiver, verdict, argument = int(receiver), int(verdict), int(argument)
            if verdict == VERDICT_ALREADY_HELD:             # relocated to the recency end
                sets[receiver].remove(argument)
                sets[receiver].append(argument) if to_end else sets[receiver].insert(0, argument)
            elif verdict == VERDICT_ACCEPTED:
                sets[receiver].append(argument) if to_end else sets[receiver].insert(0, argument)
            snapshots.append([list(s) for s in sets])
        runs.append(snapshots)
    return runs


def opinions(z, protocol: str, topology: int, alpha, arm: str = "llm") -> np.ndarray:
    """The ACT opinion of every agent at every step: the mean weight of the arguments it holds."""
    alpha = np.asarray(alpha, dtype=float)
    runs = histories(z, protocol, topology, arm)
    T = max(len(r) for r in runs)
    out = np.full((len(runs), T, int(z["N"])), np.nan)
    for r, snapshots in enumerate(runs):
        for t, snapshot in enumerate(snapshots):
            for i, held in enumerate(snapshot):
                out[r, t, i] = alpha[held].mean() if held else 0.0
    return out


def group_means(op: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean opinion of Group 1 (first half of the agents) and Group 2, averaged over runs."""
    half = op.shape[2] // 2
    return np.nanmean(op[:, :, :half], (0, 2)), np.nanmean(op[:, :, half:], (0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--topology", type=int, default=None, help="1 or 2")
    ap.add_argument("--arm", default="llm", choices=("llm", "cl"))
    ap.add_argument("--run", type=int, default=None, help="print one run's first steps")
    ap.add_argument("--alpha", default=None, help="JSON list of the model-(5) weights")
    a = ap.parse_args()

    z = load(a.space, a.model)
    print(f"{z['note']}\n")
    available = cells(z)
    print("cells:", ", ".join(f"{p} / topology {g + 1} / {arm}" for p, g, arm in available))

    protocol = a.protocol or available[0][0]
    topology = (a.topology - 1) if a.topology else available[0][1]
    runs = histories(z, protocol, topology, a.arm)
    n_steps = len(runs[0]) - 1
    print(f"\n{len(runs)} runs x {n_steps} communications, {z['N']} agents, "
          f"{z['n_args']} arguments  [{protocol}, topology {topology + 1}, {a.arm}]")

    log = z[f"log|{protocol}|topo{topology}|{a.arm}"]
    verdicts = log[..., 2][log[..., 0] >= 0]
    for name, code in (("rejected", VERDICT_REJECTED), ("accepted", VERDICT_ACCEPTED),
                       ("already held", VERDICT_ALREADY_HELD)):
        share = (verdicts == code).mean()
        print(f"  {name:13s} {share:6.1%}")

    held = np.array([[len(s) for s in snap] for snap in runs[0]])
    print(f"\nrun 0: arguments per agent {held[0].mean():.1f} at the start -> "
          f"{held[-1].mean():.1f} at the end")

    if a.alpha:
        alpha = json.loads(a.alpha)
        op = opinions(z, protocol, topology, alpha, a.arm)
        g1, g2 = group_means(op)
        print(f"group means: Group 1 {g1[0]:+.3f} -> {g1[-1]:+.3f}, "
              f"Group 2 {g2[0]:+.3f} -> {g2[-1]:+.3f}")

    if a.run is not None:
        print(f"\nfirst communications of run {a.run}:")
        for sender, receiver, verdict, argument in log[a.run][:10]:
            if sender < 0:
                break
            name = {0: "rejected", 1: "accepted", 2: "already held"}[int(verdict)]
            print(f"  agent {sender:3d} -> agent {receiver:3d}  a{argument + 1:<3d} {name}")


if __name__ == "__main__":
    main()

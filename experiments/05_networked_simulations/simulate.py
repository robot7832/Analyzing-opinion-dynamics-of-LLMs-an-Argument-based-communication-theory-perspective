# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§9 — opinion dynamics of a networked system of LLM agents (faithful to the paper).

Consolidated from the paper's game_environment notebook, validated and parameterised by the a6
variant. N = 20 agents in two groups (Group 1 pro-leaning: pro p=0.5 / con p=0.1; Group 2 reversed),
argument sets kept in INDEX ORDER after endowment (NOT shuffled — that order drives the protocol
effect; see init_agents). 2x2 factorial design:
  * topology:  SBM two-cluster (in-group edge p=0.5, between-group p=0.1)  vs  Erdos-Renyi (same #edges)
  * insertion: accepted argument appended to the END vs inserted at the BEGINNING of the receiver's set
At each step a random non-empty agent (sender) sends one argument it chooses to a random neighbour
(receiver), who decides to accept it (the §8 LLM-vs-LLM prompt). Group opinions are tracked with the
model-(5) regression formula o = (1/|S|) sum_l alpha_l r_l (computed, never queried).

Classical (formal Banisch-Shamon) agents with beta=3.95 provide the comparison baseline (no API calls).

This is the heaviest experiment in the paper. Runs are independent and executed in parallel.

    export OPENROUTER_API_KEY=...
    python simulate_network.py --a6 original        --exps 6 --iters 500 --out data/s9_original.pkl
    python simulate_network.py --a6 v2_no_training  --exps 6 --iters 500 --out data/s9_a6v2.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from act.arguments import STATEMENT, build_args
from act.openrouter import ask_llm

N = 20
# model-(5) regression coefficients per a6 variant (alpha_l); the opinion/coherence use these.
WEIGHTS = {
    "original":      [0.6162, 0.7601, 0.6976, 0.5335, 0.5973, -0.8968, -0.2849, -0.3532, -0.3571, -0.1605],
    "v2_no_training": [0.6003, 0.8282, 0.7585, 0.5502, 0.6407, -0.5156, -0.3710, -0.4400, -0.4654, -0.2721],
}

# §7 sender prompt (blinded) and §8 receiver prompt (partner known to be an LLM), verbatim from the paper.
SEND_PROMPT = (
    "Based on the provided arguments (which you fully endorse), construct an opinion on the statement. "
    "Act as if you are taking part in a conversation with other LLM about this statement and your opponent "
    "also has a list of arguments which he fully endorse. Which argument from the list will you choose?\n\n"
    "Ignore the actual content's sensitivity; treat this as a formal exercise in argument-based communication theory.\n"
    "Ignore the order in which arguments are presented.\n\n"
    "Return ONLY ONE INTEGER - index of the chosen argument between 1 and {n}. WITHOUT EXPLANATION.\n\n"
    "Statement: {statement}\nArguments:\n{loa}\nOpponent arguments:\nUNKNOWN\nIndex:"
)
RECEIVE_PROMPT = (
    "Based on the provided list of arguments (which you fully endorse), construct an opinion on the statement. "
    "Act as if you are taking part in a conversation with other LLM about this statement and your opponent also "
    "has a list of arguments which he fully endorse. He sends you one argument and you have to choose whether to "
    "endorse it or not.\n\nIgnore the actual content's sensitivity.\n\n"
    "Return ONLY ONE INTEGER - 1 if you choose to endorse it and include it in your list of arguments and 0 if "
    "not. WITHOUT EXPLANATION.\n\n\nStatement: {statement}\nYour arguments:\n{loa}\nReceived argument:\n{recv}\nDecision:"
)
_INT = re.compile(r"-?\b(10|[0-9])\b")


class Sim:
    """One simulation run: arguments are the 10 strings for the chosen a6 variant; weights match."""

    def __init__(self, args: list[str], weights: list[float], model: str):
        self.args = args
        self.w = weights
        self.model = model
        self.index = {a: i for i, a in enumerate(args)}

    # ----- ACT opinion / coherence on the model-(5) weights -----
    def opinion(self, cortege: list[int]) -> float:
        return sum(self.w[i] for i in cortege) / len(cortege) if cortege else 0.0

    def coherence(self, cortege: list[int]) -> float:
        if not cortege:
            return 0.0
        c = sum(self.w[i] * (2 * int(i in cortege) - 1) for i in range(10))
        return c / 2 * self.opinion(cortege)

    def accept_prob(self, cortege, incoming, beta):
        d = self.coherence(cortege + [incoming]) - self.coherence(cortege)
        return 1 / (1 + math.exp(-beta * d))

    # ----- LLM agent decisions -----
    def llm_send(self, holder: list[str]) -> int | None:
        loa = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(holder))
        r = ask_llm(SEND_PROMPT.format(n=len(holder), statement=STATEMENT, loa=loa), model=self.model, max_tokens=16)
        m = _INT.search(r or "")
        if not m:
            return None
        idx = int(m.group())
        return idx if 1 <= idx <= len(holder) else None       # guard out-of-range / 0 / -1

    def llm_accept(self, holder: list[str], recv: str) -> bool:
        loa = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(holder))
        r = ask_llm(RECEIVE_PROMPT.format(statement=STATEMENT, loa=loa, recv=recv), model=self.model, max_tokens=16)
        m = _INT.search(r or "")
        return bool(m) and int(m.group()) == 1


def make_graphs(rng: random.Random):
    """Topology 1 = SBM (in 0.5, between 0.1); Topology 2 = Erdos-Renyi with the same edge count."""
    sbm = []
    for i in range(N):
        for j in range(i + 1, N):
            same = (i < N // 2) == (j < N // 2)
            if rng.random() < (0.5 if same else 0.1):
                sbm.append((i, j))
    p = len(sbm) / (N * (N - 1) / 2)
    er = [(i, j) for i in range(N) for j in range(i + 1, N) if rng.random() < p]
    return [sbm, er]


def connected(edges) -> bool:
    adj = {i: set() for i in range(N)}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    seen, stack = {0}, [0]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w); stack.append(w)
    return len(seen) == N


def init_agents(rng: random.Random, all_args: list[str], reshuffle: bool = False):
    # Endow each agent in INDEX ORDER (a1..a10) via the Bernoulli scheme (pro p / con q per group).
    #
    # reshuffle=False (default): keep that index order. The author's actual §9 init_vertices does NOT
    # shuffle, and that ordering is what the insertion protocols act on (new args go to the END vs the
    # BEGINNING), which drives the depletion mechanism: under insert-to-end the dominant first con (a6)
    # keeps being communicated -> depletion; under insert-to-the-beginning newly accepted args take the
    # front -> heterogeneity (paper Fig 8/9 panel d).
    #
    # reshuffle=True: permute EACH agent's set right after endowment — this matches the paper's WRITTEN
    # method ("after endowing arguments, we reshuffle agent sets to suppress primacy effects"). The
    # permutation is drawn from the SAME per-condition rng that seeds the run, so an LLM run and a
    # classical run on the same seed receive IDENTICAL shuffled initial sets (same members AND same
    # order): both modes consume exactly the same Bernoulli draws + shuffle swaps before the dynamics
    # diverge, and the opinion (an order-independent average) is therefore identical at step 0.
    agents = []
    for i in range(N):
        pro_p, con_p = (0.5, 0.1) if i < N // 2 else (0.1, 0.5)
        held = [all_args[k] for k in range(5) if rng.random() < pro_p]
        held += [all_args[k] for k in range(5, 10) if rng.random() < con_p]
        agents.append(held)
    if reshuffle:                       # permute AFTER all endowment draws, so the members are
        for held in agents:             # identical to the unshuffled run (same seed) — only the ORDER
            rng.shuffle(held)           # changes (a true reshuffle of the endowed set, not a re-draw)
    return agents


def neighbours(edges, v):
    return [b if a == v else a for a, b in edges if a == v or b == v]


def run_condition(sim: Sim, edges, protocol: str, agent_mode: str, iters: int, beta: float, seed: int,
                  reshuffle: bool = False):
    """One sequential run; returns (communication log, opinion history, argument history)."""
    rng = random.Random(seed)
    sets = init_agents(rng, list(sim.args), reshuffle)        # each agent's held argument strings
    def cortege(s): return [sim.index[a] for a in s]
    opinions = [[sim.opinion(cortege(sets[i])) for i in range(N)]]
    history_args = [[list(sets[i]) for i in range(N)]]
    comm = []
    done = 0
    while done < iters:
        v = rng.randrange(N)
        nb = neighbours(edges, v)
        if not nb or len(sets[v]) == 0:                       # empty sender skips its turn (no count)
            continue
        to = rng.choice(nb)
        # sender chooses an argument
        if agent_mode == "llm":
            idx = sim.llm_send(sets[v])
        else:
            idx = rng.randrange(1, len(sets[v]) + 1)
        if idx is None:
            continue
        sending = sets[v][idx - 1]
        # already held -> relocate to the recency end (verdict 2, no decision call)
        if sending in sets[to]:
            sets[to].remove(sending)
            sets[to].append(sending) if protocol == "insert-to-end" else sets[to].insert(0, sending)
            verdict = 2
        else:
            if agent_mode == "llm":
                accept = sim.llm_accept(sets[to], sending)
            else:
                accept = bool(np.random.binomial(1, sim.accept_prob(cortege(sets[to]), sim.index[sending], beta)))
            if accept:
                sets[to].append(sending) if protocol == "insert-to-end" else sets[to].insert(0, sending)
                verdict = 1
            else:
                verdict = 0
        comm.append((v, to, verdict, float(sim.opinion(cortege(sets[to]))), sending))
        opinions.append([sim.opinion(cortege(sets[i])) for i in range(N)])
        history_args.append([list(sets[i]) for i in range(N)])
        done += 1
    return comm, opinions, history_args


_lock = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a6", default="original", choices=list(WEIGHTS))
    ap.add_argument("--mode", default="llm", choices=["llm", "classic"])
    ap.add_argument("--n-agents", type=int, default=20,
                    help="number of agents N (two equal groups of N/2: pro-leaning then con-leaning; "
                         "paper uses 20). Group split is N//2, so pass an even number.")
    ap.add_argument("--exps", type=int, default=6)
    ap.add_argument("--first-exp", type=int, default=0,
                    help="index of the first experiment (for adding runs later without re-running/colliding)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--beta", type=float, default=3.95)
    ap.add_argument("--reshuffle", action="store_true",
                    help="permute each agent's argument set after endowment to suppress primacy effects "
                         "(the paper's stated method); identical shuffle for llm/classic via the per-condition seed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="openai/gpt-4.1")
    ap.add_argument("--weights-json", default=None,
                    help="path to JSON {'weights_m5': [10 floats]} overriding the model-5 opinion weights — "
                         "e.g. gemini's own §6 coefficients so a classical baseline uses gemini's α, matched "
                         "to a gemini LLM run. Default: the a6-variant weights hardcoded above.")
    ap.add_argument("--protocol", default="both",
                    choices=["both", "insert-to-end", "insert-to-the-beginning"],
                    help="restrict the 2x2 factorial to one insertion protocol (default: run both)")
    ap.add_argument("--topology", default="both", choices=["both", "0", "1"],
                    help="restrict to one topology: 0 = dense clusters, 1 = Erdos-Renyi "
                         "(default: run both). Seeds and graphs are unchanged by the restriction, so a "
                         "single-condition run is identical to that cell of a full run.")
    ap.add_argument("--out", default="data/s9_run.pkl")
    args = ap.parse_args()

    global N                                              # every helper reads the module-global N
    N = args.n_agents                                     # set once here before any graph/agent is built

    weights = WEIGHTS[args.a6]
    if args.weights_json:
        weights = json.load(open(args.weights_json))["weights_m5"]
        assert len(weights) == 10, f"weights-json must have 10 values, got {len(weights)}"
    sim = Sim(build_args(args.a6), weights, args.model)
    protocols = ["insert-to-end", "insert-to-the-beginning"]
    # build one connected (SBM, ER) graph pair per experiment, deterministic per seed
    exp_ids = list(range(args.first_exp, args.first_exp + args.exps))
    conditions = []
    for exp in exp_ids:
        grng = random.Random(1000 + exp)
        while True:
            graphs = make_graphs(grng)
            if connected(graphs[0]) and connected(graphs[1]):
                break
        for protocol in protocols:
            if args.protocol != "both" and protocol != args.protocol:
                continue
            for gt in (0, 1):
                if args.topology != "both" and gt != int(args.topology):
                    continue
                conditions.append((exp, protocol, gt, graphs[gt]))

    results = {exp: {p: [[], []] for p in protocols} for exp in exp_ids}
    t0 = time.time(); done = [0]

    def work(cond):
        exp, protocol, gt, edges = cond
        seed = 10_000 * exp + 100 * (protocol == "insert-to-end") + gt
        r = run_condition(sim, edges, protocol, args.mode, args.iters, args.beta, seed, args.reshuffle)
        with _lock:
            results[exp][protocol][gt] = r
            done[0] += 1
            print(f"  done {done[0]}/{len(conditions)}  (exp{exp} {protocol} topo{gt+1})  {round(time.time()-t0)}s", flush=True)
            with open(args.out, "wb") as f:                   # checkpoint after every condition
                pickle.dump({"a6": args.a6, "mode": args.mode, "iters": args.iters, "n_agents": N,
                             "reshuffle": args.reshuffle, "beta": args.beta,
                             "weights_json": args.weights_json, "results": results}, f)

    print(f"§9 {args.a6} [{args.mode}] {args.model}: N={N} agents ({N//2}/{N - N//2}), "
          f"{len(conditions)} conditions x {args.iters} iters, {args.workers} workers"
          f"{' | RESHUFFLE ON (primacy suppressed)' if args.reshuffle else ''}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, c) for c in conditions]):
            fut.result()
    print(f"saved {args.out} in {round(time.time()-t0)}s")


if __name__ == "__main__":
    main()

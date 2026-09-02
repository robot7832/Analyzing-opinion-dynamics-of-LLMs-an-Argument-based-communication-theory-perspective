# Argument-based communication in large language models — code and data

This repository holds the code and data behind the paper's experiments. It is an anonymous
artifact: it deliberately carries no author names, affiliations, locations, e-mail addresses or
credentials, and its git history contains a single commit by an anonymous author.

Language models are queried through OpenRouter. Four argument spaces and three models are covered:

| symbol | argument space |
|---|---|
| `A_gc` | 10 arguments on gun policy (the paper's initial space) |
| `A_-6` | the same 10, with argument a6 replaced |
| `A_gc+` | 16 arguments on gun policy (enlarged space) |
| `A_cc` | 10 arguments on calorie-count labelling (second topic) |

Models: `gpt-4.1`, `gemini-2.5-flash-lite`, `claude-sonnet-4`.

## How the repository is organised

Four levels, in this order:

1. **experiment** — numerical opinion, sending, receiving, epistemic networks, networked
   simulations, semantic independence;
2. **shared code of that experiment** — the protocol, the data generation and the plotting, which
   are identical across spaces and models and therefore live once, at the experiment level;
3. **argument space** — `A_gc`, `A_-6`, `A_gc+`, `A_cc`;
4. **model** — `gpt-4.1`, `gemini-2.5-flash-lite`, `claude-sonnet-4`.

Two rules hold throughout. **Data generation and plotting are separate files**: `collect*.py`
queries the models and costs money, `plot_*.py` only reads what is already stored and is free.
**Data lives in its own top-level branch**, `data/`, mirroring the same four levels, so the code
tree stays readable and the data can be inspected without reading any code. Figures are written to
`figures/`, mirroring the same hierarchy again. There are no notebooks: every entry point is a
`.py` file.

## Tree

```
|-- common/
|   |-- openrouter.py
|   |-- spaces.py
|   `-- style.py
|-- data/
|   |-- 01_numerical_opinion/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (3 files)
|   |   |   `-- gpt-4.1/   (4 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (4 files)
|   |   |   |-- gemini-2.5-flash-lite/   (4 files)
|   |   |   `-- gpt-4.1/   (4 files)
|   |   |-- A_gc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (3 files)
|   |   |   `-- gpt-4.1/   (6 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (4 files)
|   |       `-- gpt-4.1/   (4 files)
|   |-- 02_sending/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (2 files)
|   |   |   `-- gpt-4.1/   (2 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (2 files)
|   |   |   |-- gemini-2.5-flash-lite/   (2 files)
|   |   |   `-- gpt-4.1/   (2 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (3 files)
|   |   |   `-- gpt-4.1/   (8 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (1 files)
|   |       `-- gpt-4.1/   (1 files)
|   |-- 03_receiving/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (1 files)
|   |       `-- gpt-4.1/   (1 files)
|   |-- 04_epistemic_networks/
|   |   `-- A_gc/
|   |       `-- gpt-4.1/   (1 files)
|   |-- 05_networked_simulations/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (3 files)
|   |   |   `-- gpt-4.1/   (3 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (2 files)
|   |   |   |-- gemini-2.5-flash-lite/   (2 files)
|   |   |   `-- gpt-4.1/   (2 files)
|   |   `-- A_gc/
|   |       |-- gemini-2.5-flash-lite/   (3 files)
|   |       `-- gpt-4.1/   (3 files)
|   `-- 06_semantic_independence/
|       |-- A_cc/
|       |   `-- pairwise_calibrated_sts.csv
|       `-- A_gc/
|           |-- independence_test_results.csv
|           `-- pairwise_calibrated_sts.csv
|-- experiments/
|   |-- 01_numerical_opinion/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc+/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- collect.py
|   |   |-- collect_order.py
|   |   |-- plot_order_tables.py
|   |   `-- plot_regression.py
|   |-- 02_sending/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc+/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- analyse.py
|   |   |-- collect.py
|   |   |-- collect_pair.py
|   |   `-- plot_cortege_ci.py
|   |-- 03_receiving/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc+/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- collect.py
|   |   |-- fit_beta.py
|   |   `-- plot_acceptance_heatmap.py
|   |-- 04_epistemic_networks/
|   |   |-- A_gc/
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- analyse.py
|   |   |-- collect.py
|   |   `-- plot_persuasiveness.py
|   |-- 05_networked_simulations/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- aggregate.py
|   |   |-- plot_opinion_dynamics.py
|   |   |-- plot_topologies.py
|   |   |-- reconstruct.py
|   |   `-- simulate.py
|   `-- 06_semantic_independence/
|       |-- A_-6/
|       |   `-- run.py
|       |-- A_cc/
|       |   `-- run.py
|       |-- A_gc/
|       |   `-- run.py
|       |-- A_gc+/
|       |   `-- run.py
|       |-- analyse.py
|       `-- plot_similarity.py
|-- figures/
|   |-- 01_numerical_opinion/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (1 files)
|   |       `-- gpt-4.1/   (1 files)
|   |-- 02_sending/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (1 files)
|   |       `-- gpt-4.1/   (1 files)
|   |-- 03_receiving/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (1 files)
|   |       `-- gpt-4.1/   (1 files)
|   |-- 04_epistemic_networks/
|   |   `-- A_gc/
|   |       `-- gpt-4.1/   (1 files)
|   |-- 05_networked_simulations/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (1 files)
|   |   |   |-- gemini-2.5-flash-lite/   (1 files)
|   |   |   `-- gpt-4.1/   (1 files)
|   |   `-- A_gc/
|   |       |-- gemini-2.5-flash-lite/   (1 files)
|   |       `-- gpt-4.1/   (1 files)
|   `-- 06_semantic_independence/
|       |-- A_-6/
|       |   `-- .gitkeep
|       |-- A_cc/
|       |   `-- .gitkeep
|       |-- A_gc/
|       |   `-- .gitkeep
|       `-- A_gc+/
|           `-- .gitkeep
|-- spaces/
|   |-- A_-6.json
|   |-- A_cc.json
|   |-- A_gc+.json
|   `-- A_gc.json
|-- .gitignore
|-- README.md
|-- coverage.json
`-- requirements.txt
```

## How to use

### Install

```bash
pip install -r requirements.txt
```

Everything except the semantic-independence experiment runs on those packages alone. That one
additionally needs `sentence-transformers` and downloads two encoders on first use.

### The two steps

Every experiment is split in two, and the split is the thing to understand before running anything:

- **`collect*.py` queries the models.** It needs `OPENROUTER_API_KEY` in the environment, it needs
  network access, and it **costs money**. It writes into `data/`.
- **`plot_*.py`, `analyse.py`, `fit_beta.py` and `reconstruct.py` read what is already in `data/`.**
  They need no key, no network and no money. They write into `figures/` or print to the terminal.

Everything in `data/` is already collected, so every figure and every number in the paper can be
rebuilt without spending anything. Re-collection is only needed to extend the study to a new model,
a new space, or a larger number of trials.

Each leaf carries a `run.py` that wires its own space and model into the experiment's scripts:

```bash
cd experiments/01_numerical_opinion/A_gc/gpt-4.1
python run.py --plot                 # free
python run.py --collect              # queries the model, COSTS MONEY
```

The experiment-level scripts can also be called directly, which is what you want when you need
flags the leaf runner does not pass through:

```bash
cd experiments/01_numerical_opinion
python collect_order.py --space A_gc --model openai/gpt-4.1 --trials 30 --only t4
```

### Which script does what

| experiment | what it measures | generation | reading and plotting |
|---|---|---|---|
| `01_numerical_opinion` | the opinion a model reports for every subset of the space, and how much the order of the arguments moves it | `collect.py` (all subsets), `collect_order.py` (the order experiments) | `plot_regression.py` (the weight estimates), `plot_order_tables.py` (order tables and their LaTeX) |
| `02_sending` | which argument a model chooses to send, and whether that choice depends on the partner and on the order of its own set | `collect.py` (four corteges x three partner settings), `collect_pair.py` (the two-argument contrast), `collect_permuted.py` where present | `plot_cortege_ci.py` (frequencies with Wilson intervals), `analyse.py` |
| `03_receiving` | whether a model adopts an incoming argument, as a function of what it already holds | `collect.py` (every ordered pair of arguments x three settings) | `plot_acceptance_heatmap.py` (the acceptance atlases), `fit_beta.py` (the biased-processing coefficient) |
| `04_epistemic_networks` | how persuasive each argument is on its own, rated by several models and by human raters | `collect.py` | `plot_persuasiveness.py`, `analyse.py` |
| `05_networked_simulations` | what happens when the agents talk to each other on a network | `simulate.py` (the runs), `aggregate.py` (raw runs to aggregates) | `reconstruct.py` (full per-agent dynamics), `plot_opinion_dynamics.py`, `plot_topologies.py` |
| `06_semantic_independence` | how distinct the arguments of a space are from each other | `encode.py` | `plot_similarity.py`, `analyse.py` |

### Worked examples

Rebuild the weight estimates of one model on one space, and the order tables that go with them:

```bash
cd experiments/01_numerical_opinion
python plot_regression.py --space A_gc --model gpt-4.1
python plot_order_tables.py                      # prints the table and its LaTeX
```

Rebuild the sending figure for a permuted cortege, solid for the original order and dashed for the
permuted one:

```bash
cd experiments/02_sending
python plot_cortege_ci.py --model claude          # or gpt, gemini
```

Rebuild the acceptance atlases and refit the biased-processing coefficient:

```bash
cd experiments/03_receiving
python plot_acceptance_heatmap.py
python fit_beta.py --space A_gc --model gpt-4.1
```

Read the simulations. This is the entry point for anything the aggregates do not already answer,
because it returns the opinion of every agent at every step rather than group means:

```bash
cd experiments/05_networked_simulations
python reconstruct.py --space A_gc --model gpt-4.1 --protocol insert-to-end --topology 2
python reconstruct.py --space A_cc --model claude-sonnet-4 --run 0    # first steps of one run
```

```python
import reconstruct as R
z = R.load("A_gc", "gpt-4.1")
op = R.opinions(z, "insert-to-end", 1, alpha)   # (runs, steps, agents)
g1, g2 = R.group_means(op)
```

Redraw the opinion trajectories and the two topologies:

```bash
cd experiments/05_networked_simulations
python plot_opinion_dynamics.py --space A_gc --model gpt-4.1
python plot_topologies.py
```

Re-run the simulations themselves, which is the one expensive step here — a full cell is 500
model-driven communications per run:

```bash
export OPENROUTER_API_KEY="..."
cd experiments/05_networked_simulations
python simulate.py --space A_gc --model openai/gpt-4.1 --exps 5 --iters 500
python aggregate.py                              # raw runs -> s9_agg.npz
```

### Things that apply everywhere

- **Temperature is 1 in every query.** Note that 1 is the maximum the Anthropic API accepts, while
  for the OpenAI and Google models it is the default and half of their maximum, so the same nominal
  value is not the same sampling regime across model families.
- **Answers are sampled, not deterministic.** Re-collecting will not reproduce a stored table cell
  for cell. Where a single reading is unstable, the tables carry a modal or mean value over repeated
  queries and the per-query readings are kept next to them, so the spread can be inspected rather
  than guessed at.
- **Spend deliberately.** The cost of a collection scales with the number of subsets, corteges or
  simulation steps, and the simulations are by far the most expensive part. Start with a small
  `--exps` and check the output before launching a full cell.
- **Not every combination exists.** `coverage.json` lists the (experiment, space, model) triples
  that were actually collected; the missing leaves are absent rather than empty.

## Opinion dynamics: how the simulations work and what the stored arrays mean

`05_networked_simulations` is the only experiment where the models do not answer a question about a
fixed argument set — they talk to each other and the sets change.

**One run.** `N` agents sit on a fixed graph. Each agent starts with a set of arguments drawn by a
Bernoulli scheme: the first half of the agents (Group 1) draw each pro argument with probability 0.5
and each con argument with probability 0.1, the second half (Group 2) the other way round, which
makes the two groups start on opposite sides. At every step one agent is picked at random, chooses
*one* argument from its own set to send, and a random neighbour decides whether to adopt it. Both
decisions are the model's: the sender is given the sending prompt of `02_sending`, the receiver the
acceptance prompt of `03_receiving`. Only the receiver's set changes, so a single communication is
one-directional; roles are redrawn every step, so over a run a pair exchanges in both directions.
A run is 500 accepted-or-rejected communications.

**Two factors, crossed.** *Topology* — Topology 1 is a two-cluster stochastic block model (in-group
edge probability 0.5, between-group 0.1); Topology 2 is an Erdos-Renyi graph with the same edge
count, so the two differ in modularity, not in density. *Insertion protocol* — an adopted argument
is appended to the end of the receiver's set, or inserted at its beginning. The protocol matters
because agents read their arguments in order, so it decides which argument sits in first position.

**The opinion is computed, never queried.** No agent is ever asked for a number during a run. An
agent's opinion is the mean of the regression weights of the arguments it currently holds,
`o_i = (1 / |S_i|) * sum over l in S_i of alpha_l`, with `alpha` taken from the model-(5) regression
fitted to *that same model on that same space* in `01_numerical_opinion`. This matters when reading
the figures: a trajectory is the behaviour of one model read through the ruler of one model, and the
ruler is a choice. Changing whose `alpha` is used shifts the levels of the curves substantially while
leaving their shape and the ordering of the models intact, so level comparisons across models are
only meaningful once the same `alpha` is used for all of them.

**The classical baseline.** Every LLM run is paired with a run of formal ACT agents on the same
graph and the same initial endowment, differing only in the decision rules: a classical sender picks
uniformly at random, and a classical receiver accepts with probability `sigma(beta * delta C)`, where
`delta C` is the change in coherence and `beta` is fitted to that model's own acceptance data in
`03_receiving`. The pair shares its seed, so at step 0 the two arms are identical and any divergence
is caused by the decision rules alone. Note that an argument the receiver already holds is not put
to the model at all — it is moved to the recency end of the set — so the diagonal of an acceptance
matrix never enters the dynamics.

**Agents may start empty.** With these probabilities a few agents draw no arguments at all. Such an
agent can still accept, but cannot send: its turn is skipped and not counted towards the 500.

**What is stored, and why it is small.** The simulator's own output keeps, at every step, a snapshot
of every agent's ordered argument set. Across all collections that is about 2.1 GB, with single
files past the platform's 100 MB limit. Nearly all of it is redundant: those snapshots are a
deterministic function of two much smaller things — the ordered initial sets and the communication
log — once the insertion protocol is known. `runs.npz` stores those two, which is lossless and about
280 times smaller; `experiments/05_networked_simulations/reconstruct.py` replays the log and returns
the snapshots and the per-agent opinion trajectories. The reconstruction was checked against the
original snapshots step for step, ordering included, and the group-mean trajectories it produces
match the stored aggregates to 3e-16.

`runs.npz` keys, with `<arm>` either `llm` for the model-driven agents or `cl` for their matched
classical partners:

- `exp_ids` `(n_runs,)` — the experiment id of each run, which is also its random seed offset;
- `edges|topo<0|1>` `(n_runs, E, 2)` — the graph of each run, an edge list padded with `-1`;
- `init|<protocol>|topo<0|1>|<arm>` `(n_runs, N, n_args)` — the initial sets as argument indices
  **in the order the agent holds them**, padded with `-1`. The order matters: it is what the
  insertion protocol acts on;
- `log|<protocol>|topo<0|1>|<arm>` `(n_runs, T, 4)` — one row per communication: sender, receiver,
  verdict, argument index. The verdict is `0` rejected, `1` accepted, `2` the receiver already held
  it and moved it to the recency end without consulting the model.

Reading one cell:

```bash
cd experiments/05_networked_simulations
python reconstruct.py --space A_gc --model gpt-4.1 --protocol insert-to-end --topology 2
```

which reports the accept/reject/already-held breakdown and how the sets grow, and with `--alpha`
also the group-mean opinion trajectories. `histories()` and `opinions()` are importable if you want
the arrays rather than the summary.

**What is in `s9_agg.npz`.** The same runs pre-averaged, which is what the figures read directly.
Keys are strings:

- `<protocol>|topo<0|1>|<llm|cl>|g<1|2>_mean` — the group-mean opinion trajectory, averaged over
  runs, where `cl` is the classical arm;
- `... |g<1|2>_ci` — the half-width of the 95 % confidence interval of that mean across runs (a
  confidence interval, not a standard deviation: at n = 30 the two differ by a factor of about 2.7);
- `argdist|<protocol>|topo<0|1>|<llm|cl>` — a (time window x n arguments) table with the percentage
  share of each argument among the transmissions in that window, averaged over runs;
- `n_runs`, `N`, `window`, `note` — how many matched runs the aggregate covers, the number of
  agents, the window length in iterations, and a short provenance string.

One window equals one communication per agent, so it is 20 iterations at `N = 20` and 100 at
`N = 100`. The gun-topic collections use `N = 20` for `gpt-4.1` and `N = 100` for
`gemini-2.5-flash-lite`; the calorie collections use `N = 20` throughout. Entropies and other
sample-size-sensitive statistics are therefore not comparable across collections unless the
estimator's sample size is equalised first.

Only runs present in *both* arms are aggregated, so an LLM run whose classical partner is missing is
dropped rather than compared against nothing.

## What is in `data/`

| experiment | space | model | files |
|---|---|---|---|
| 01_numerical_opinion | `A_gc` | `gpt-4.1` | opinions.csv · opinions_shuffled.csv · order_table3.csv · order_table4.csv · order_table4_t30.csv · order_trials_t30.csv |
| 01_numerical_opinion | `A_gc` | `gemini-2.5-flash-lite` | opinions.csv · order_table3.csv · order_table4.csv |
| 01_numerical_opinion | `A_gc` | `claude-sonnet-4` | opinions.csv |
| 01_numerical_opinion | `A_-6` | `gpt-4.1` | opinions.csv · opinions_shuffled.csv · order_table3.csv · order_table4.csv |
| 01_numerical_opinion | `A_-6` | `gemini-2.5-flash-lite` | opinions.csv · order_table3.csv · order_table4.csv |
| 01_numerical_opinion | `A_gc+` | `gpt-4.1` | opinions.csv · opinions_shuffled.csv · order_table3.csv · order_table4.csv |
| 01_numerical_opinion | `A_gc+` | `gemini-2.5-flash-lite` | opinions.csv · opinions_shuffled.csv · order_table3.csv · order_table4.csv |
| 01_numerical_opinion | `A_cc` | `gpt-4.1` | opinions.csv · opinions_shuffled.csv · order_table3.csv · order_table4.csv |
| 01_numerical_opinion | `A_cc` | `gemini-2.5-flash-lite` | opinions.csv · opinions_shuffled.csv · order_table3.csv · order_table4.csv |
| 01_numerical_opinion | `A_cc` | `claude-sonnet-4` | opinions.csv · opinions_shuffled.csv · order_table3.csv · order_table4.csv |
| 02_sending | `A_gc` | `gpt-4.1` | cortege_a10_a8_a6.csv · cortege_a1_a3_a5.csv · cortege_a5_a10_a8.csv · sending.csv · sending_pair_T10.csv · sending_pair_T50.csv · sending_pair_T50_llm.csv · sending_permuted.csv |
| 02_sending | `A_gc` | `gemini-2.5-flash-lite` | sending.csv · sending_pair_T50.csv · sending_permuted.csv |
| 02_sending | `A_-6` | `gpt-4.1` | sending.csv · sending_permuted.csv |
| 02_sending | `A_-6` | `gemini-2.5-flash-lite` | sending.csv · sending_permuted.csv |
| 02_sending | `A_gc+` | `gpt-4.1` | sending.csv |
| 02_sending | `A_gc+` | `gemini-2.5-flash-lite` | sending.csv |
| 02_sending | `A_cc` | `gpt-4.1` | sending.csv · sending_permuted.csv |
| 02_sending | `A_cc` | `gemini-2.5-flash-lite` | sending.csv · sending_permuted.csv |
| 02_sending | `A_cc` | `claude-sonnet-4` | sending.csv · sending_permuted.csv |
| 03_receiving | `A_gc` | `gpt-4.1` | receiving.csv |
| 03_receiving | `A_gc` | `gemini-2.5-flash-lite` | receiving.csv |
| 03_receiving | `A_-6` | `gpt-4.1` | receiving.csv |
| 03_receiving | `A_-6` | `gemini-2.5-flash-lite` | receiving.csv |
| 03_receiving | `A_gc+` | `gpt-4.1` | receiving.csv |
| 03_receiving | `A_gc+` | `gemini-2.5-flash-lite` | receiving.csv |
| 03_receiving | `A_cc` | `gpt-4.1` | receiving.csv |
| 03_receiving | `A_cc` | `gemini-2.5-flash-lite` | receiving.csv |
| 03_receiving | `A_cc` | `claude-sonnet-4` | receiving.csv |
| 04_epistemic_networks | `A_gc` | `gpt-4.1` | persuasiveness_scores.csv |
| 05_networked_simulations | `A_gc` | `gpt-4.1` | fig10.npz · runs.npz · s9_agg.npz |
| 05_networked_simulations | `A_gc` | `gemini-2.5-flash-lite` | fig10.npz · runs.npz · s9_agg.npz |
| 05_networked_simulations | `A_-6` | `gpt-4.1` | fig10.npz · runs.npz · s9_agg.npz |
| 05_networked_simulations | `A_-6` | `gemini-2.5-flash-lite` | fig10.npz · runs.npz · s9_agg.npz |
| 05_networked_simulations | `A_cc` | `gpt-4.1` | runs.npz · s9_agg.npz |
| 05_networked_simulations | `A_cc` | `gemini-2.5-flash-lite` | runs.npz · s9_agg.npz |
| 05_networked_simulations | `A_cc` | `claude-sonnet-4` | runs.npz · s9_agg.npz |
| 06_semantic_independence | `A_gc` | n/a | independence_test_results.csv · pairwise_calibrated_sts.csv |
| 06_semantic_independence | `A_-6` | n/a | - |
| 06_semantic_independence | `A_gc+` | n/a | - |
| 06_semantic_independence | `A_cc` | n/a | pairwise_calibrated_sts.csv |

File conventions:

- `opinions.csv` — one row per argument subset, columns `1..n` are 0/1 membership indicators and
  `opinion` is the model's 1-7 rating (`-1` marks an unparseable answer);
- `opinions_shuffled.csv` — the same, with each argument string presented in a random order, used
  as the control for primacy;
- `order_table3.csv`, `order_table4.csv` — the order experiments: `o_pc` reads the pro
  argument(s) first, `o_cp` reads the con argument(s) first, `diff` is their difference;
- `order_trials_t30.csv` — every individual reading behind a 30-query cell, so cell-level
  uncertainty can be recomputed;
- `sending.csv` — one row per decision, columns `cortege`, `setting` (human / llm / blinded),
  `opp_valence`, `chosen`; `sending_permuted.csv` is the same cortege read in reverse order;
- `receiving.csv` — one row per decision, the receiver's own argument, the incoming argument, the
  setting and the binary outcome;
- `runs.npz` — the complete record of every simulation run, losslessly packed (see below);
- `s9_agg.npz` — the aggregated simulation output every figure of the networked section reads:
  mean group-opinion trajectories with confidence bands, and the per-window distribution of
  transmitted arguments, for each protocol and topology, for LLM and classical agents;
- `persuasiveness_scores.csv` — prior persuasiveness ratings, 1-7. Human raters appear as
  `human_1 ... human_4`; their identities are not part of this artifact.

## What is deliberately absent

- **The simulator's raw pickles.** Not their content — `runs.npz` carries every run losslessly, as
  explained above — only the redundant per-step snapshots, which `reconstruct.py` rebuilds exactly.
  `simulate.py` regenerates the pickles from scratch if they are wanted in their original form.
- **Credentials.** No key is stored anywhere; `OPENROUTER_API_KEY` is read from the environment.
- **Identifying material.** Author names, affiliations, locations and local filesystem paths were
  removed, and the build refuses to produce this repository if any reappear.

## Coverage gaps

`claude-sonnet-4` was run on `A_gc` for the numerical-opinion experiment only, so its sending,
receiving and simulation leaves do not exist for that space. `A_gc+` has no networked simulations.
`coverage.json` lists exactly which (experiment, space, model) combinations exist.

## Scale

61 Python files, 73 CSV tables, 18 compressed arrays, 12 MB in total.

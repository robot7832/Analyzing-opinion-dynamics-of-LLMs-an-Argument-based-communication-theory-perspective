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
- `runs.npz` — the complete record of every simulation run, losslessly packed: the ordered initial
  argument sets and the communication log, from which `reconstruct.py` replays every per-step
  snapshot and per-agent opinion trajectory;
- `s9_agg.npz` — the aggregated simulation output every figure of the networked section reads:
  mean group-opinion trajectories with confidence bands, and the per-window distribution of
  transmitted arguments, for each protocol and topology, for LLM and classical agents;
- `persuasiveness_scores.csv` — prior persuasiveness ratings, 1-7. Human raters appear as
  `human_1 ... human_4`; their identities are not part of this artifact.

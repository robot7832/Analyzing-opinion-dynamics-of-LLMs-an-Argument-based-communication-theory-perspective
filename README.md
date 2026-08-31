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
|   |   |   |-- gemini-2.5-flash-lite/   (2 files)
|   |   |   `-- gpt-4.1/   (2 files)
|   |   `-- A_gc/
|   |       |-- gemini-2.5-flash-lite/   (2 files)
|   |       `-- gpt-4.1/   (2 files)
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
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (0 files)
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   |-- A_gc/
|   |   |   |-- claude-sonnet-4/   (0 files)
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (0 files)
|   |       `-- gpt-4.1/   (0 files)
|   |-- 02_sending/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (0 files)
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (0 files)
|   |       `-- gpt-4.1/   (0 files)
|   |-- 03_receiving/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (0 files)
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   |-- A_gc/
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   `-- A_gc+/
|   |       |-- gemini-2.5-flash-lite/   (0 files)
|   |       `-- gpt-4.1/   (0 files)
|   |-- 04_epistemic_networks/
|   |   `-- A_gc/
|   |       `-- gpt-4.1/   (0 files)
|   |-- 05_networked_simulations/
|   |   |-- A_-6/
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   |-- A_cc/
|   |   |   |-- claude-sonnet-4/   (0 files)
|   |   |   |-- gemini-2.5-flash-lite/   (0 files)
|   |   |   `-- gpt-4.1/   (0 files)
|   |   `-- A_gc/
|   |       |-- gemini-2.5-flash-lite/   (0 files)
|   |       `-- gpt-4.1/   (0 files)
|   `-- 06_semantic_independence/
|       |-- A_-6/
|       |-- A_cc/
|       |-- A_gc/
|       `-- A_gc+/
|-- spaces/
|   |-- A_-6.json
|   |-- A_cc.json
|   |-- A_gc+.json
|   `-- A_gc.json
`-- coverage.json
```

## Running

Nothing needs to be installed beyond the requirements:

```bash
pip install -r requirements.txt
```

Every leaf is self-contained. Rebuilding a figure from the stored data needs no key and no network:

```bash
cd experiments/01_numerical_opinion/A_gc/gpt-4.1
python run.py --plot
```

Re-collecting the data queries the model and **costs money**:

```bash
export OPENROUTER_API_KEY="..."      # never stored in the repository
cd experiments/01_numerical_opinion/A_gc/gpt-4.1
python run.py --collect
```

All model queries use temperature 1. Note that 1 is the maximum the Anthropic API accepts, while
for the OpenAI and Google models it is the default and half of their maximum, so the nominal value
is not the same sampling regime across model families.

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
| 05_networked_simulations | `A_gc` | `gpt-4.1` | fig10.npz · s9_agg.npz |
| 05_networked_simulations | `A_gc` | `gemini-2.5-flash-lite` | fig10.npz · s9_agg.npz |
| 05_networked_simulations | `A_-6` | `gpt-4.1` | fig10.npz · s9_agg.npz |
| 05_networked_simulations | `A_-6` | `gemini-2.5-flash-lite` | fig10.npz · s9_agg.npz |
| 05_networked_simulations | `A_cc` | `gpt-4.1` | - |
| 05_networked_simulations | `A_cc` | `gemini-2.5-flash-lite` | - |
| 05_networked_simulations | `A_cc` | `claude-sonnet-4` | - |
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
- `s9_agg.npz` — the aggregated simulation output every figure of the networked section reads:
  mean group-opinion trajectories with confidence bands, and the per-window distribution of
  transmitted arguments, for each protocol and topology, for LLM and classical agents;
- `persuasiveness_scores.csv` — prior persuasiveness ratings, 1-7. Human raters appear as
  `human_1 ... human_4`; their identities are not part of this artifact.

## What is deliberately absent

- **Raw simulation pickles.** The full per-step histories of the networked simulations come to
  about 2.2 GB, and seven single files exceed the platform's 100 MB limit. `data/` therefore keeps
  the aggregates (`s9_agg.npz`), which is what every figure actually reads, and
  `experiments/05_networked_simulations/simulate.py` plus `aggregate.py` regenerate the raw
  histories and the aggregates from scratch.
- **Credentials.** No key is stored anywhere; `OPENROUTER_API_KEY` is read from the environment.
- **Identifying material.** Author names, affiliations, locations and local filesystem paths were
  removed, and the build refuses to produce this repository if any reappear.

## Coverage gaps

`claude-sonnet-4` was run on `A_gc` for the numerical-opinion experiment only, so its sending,
receiving and simulation leaves do not exist for that space. `A_gc+` has no networked simulations.
`coverage.json` lists exactly which (experiment, space, model) combinations exist.

## Scale

60 Python files, 73 CSV tables, 8 compressed arrays, 3 MB in total.

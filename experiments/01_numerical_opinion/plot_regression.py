# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§6.1 + §6.3 — regress the LLM's numerical opinion on the argument set.

Two models (opinions projected to [-0.5, 0.5]; no intercept; uncentered R-squared):
    model (4):  o_hat = sum_l alpha_l r_l                  (§6.1)
    model (5):  o_hat = (1/|S|) sum_l alpha_l r_l           (§6.3, length-normalized)

Reproduces Figure 1 (panel a/d/f, model 4) and Figure 2 (model 4 vs 5) for the three
models whose opinion tables ship in data/. No API calls — pure re-analysis.

    python regression.py
"""
from __future__ import annotations

from pathlib import Path

from act.arguments import LABELS, VALENCE
from act.opinion_models import dominant_con, fit_opinion_regression, load_opinions

DATA = Path(__file__).parent / "data"
MODELS = [
    ("gpt-4.1", "opinions_gpt.csv"),
    ("claude-sonnet-4", "opinions_claude.csv"),
    ("gemini-2.5-flash-lite", "opinions_gemini.csv"),
]


def report(model_name: str, csv: str) -> None:
    df = load_opinions(DATA / csv, paper_format=True)
    m4 = fit_opinion_regression(df, length_normalized=False)
    m5 = fit_opinion_regression(df, length_normalized=True)

    print(f"\n{'=' * 64}\n{model_name}  (N = {len(df)} non-empty argument sets)\n{'=' * 64}")
    print(f"{'arg':<5}{'valence':<9}{'model(4)':>10}{'model(5)':>10}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:<5}{VALENCE[i]:<9}{m4.params[lab]:>+10.3f}{m5.params[lab]:>+10.3f}")
    print(f"{'-' * 34}")
    print(f"uncentered R^2          {m4.rsquared:>10.3f}{m5.rsquared:>10.3f}")
    print(f"pro coefficients > 0:   model4={all(m4.params[LABELS[i]] > 0 for i in range(5))}  "
          f"model5={all(m5.params[LABELS[i]] > 0 for i in range(5))}")
    print(f"dominant con argument:  model4={dominant_con(m4.params.values)}  "
          f"model5={dominant_con(m5.params.values)}")


def main() -> None:
    for model_name, csv in MODELS:
        if (DATA / csv).exists():
            report(model_name, csv)
    print("\nTakeaway: pro arguments get positive weights, con negative; a6 has the largest")
    print("|coefficient| among con args; model (5) raises the uncentered R^2 (length matters).")


if __name__ == "__main__":
    main()

# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§8 — analyze the endorsement data: biased processing and the Banisch-Shamon beta.

Reads either the paper's own table (columns model_argument, received_argument, type, verdict)
or a fresh collect_receiving.py output (held, received, setting, endorse). Reports:
  * acceptance of same- vs opposite-valence arguments per sender setting (biased processing),
  * the exceptional acceptance of argument a4,
  * the biased-processing strength beta per setting (eq. 3 & 6), using the model (5)
    coefficients fitted on the §6 opinion data.

No API calls.

    python analyze_receiving.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from act.arguments import LABELS, VALENCE
from act.opinion_models import estimate_beta, fit_opinion_regression, load_opinions

HERE = Path(__file__).parent
S6_OPINIONS = HERE.parent / "s6_arguments_to_opinions" / "data" / "opinions_gpt.csv"


def load_receiving(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model_argument" in df.columns:                       # paper format
        df = df.rename(columns={"model_argument": "held_i", "received_argument": "received_i",
                                "type": "setting", "verdict": "endorse"})
    else:                                                     # collect_receiving.py format
        df["held_i"] = df["held"].map(LABELS.index)
        df["received_i"] = df["received"].map(LABELS.index)
    df["held_val"] = df["held_i"].map(lambda i: VALENCE[i])
    df["received_val"] = df["received_i"].map(lambda i: VALENCE[i])
    return df


def main() -> None:
    data_path = HERE / "data" / "receiving_gpt.csv"
    df = load_receiving(data_path)

    print("Acceptance of same- vs opposite-valence arguments (biased processing):")
    for setting in ("human", "llm", "unknown"):
        s = df[df.setting == setting]
        same = s[s.held_val == s.received_val].endorse.mean()
        opp = s[s.held_val != s.received_val].endorse.mean()
        print(f"  {setting:<8} same-valence={same * 100:4.0f}%   opposite-valence={opp * 100:4.0f}%")

    a4 = df[df.received_i == 3].endorse.mean()
    print(f"\nArgument a4 received: accepted {a4 * 100:.0f}% (the exceptional argument, not a6).")

    # beta per setting from the §6 model (5) coefficients
    alpha = fit_opinion_regression(load_opinions(S6_OPINIONS, paper_format=True), True).params.values
    print("\nBiased-processing beta (eq. 3 & 6, model-5 coefficients):")
    for setting, label in (("human", "human"), ("llm", "llm"), ("unknown", "blinded")):
        cells = (df[df.setting == setting].groupby(["held_i", "received_i"]).endorse
                 .agg(n_accept="sum", n_total="count").reset_index()
                 .rename(columns={"held_i": "held", "received_i": "received"}))
        beta = estimate_beta(cells, alpha)
        print(f"  {label:<8} beta = {beta:.2f}")
    print("  (paper, original a6: human 4.18 / llm 3.95 / blinded 3.75; all >> Banisch-Shamon 0.25-0.70)")


if __name__ == "__main__":
    main()

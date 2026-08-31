# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§7 — selection frequencies per cortege (Figure 4).

Reads the paper's aggregated frequency tables (data/cortege_*.csv): for each cortege and
each sender setting (human / llm / unknown) and receiver valence, the fraction of trials
each argument was chosen to be sent. Highlights cortege (a10, a8, a6), where the model
overwhelmingly sends a6 regardless of the partner.

No API calls.

    python analyze_sending.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
CORTEGES = {
    "(a1, a3, a5)": "cortege_a1_a3_a5.csv",
    "(a5, a10, a8)": "cortege_a5_a10_a8.csv",
    "(a10, a8, a6)": "cortege_a10_a8_a6.csv",
}


def show(name: str, csv: str) -> None:
    df = pd.read_csv(csv)
    df["arg"] = df["argument"].str.replace("_", "")
    print(f"\n{name}  — selection frequency by setting and receiver valence")
    print(f"  {'arg':<5}{'human/neg':>10}{'human/pos':>10}{'llm/neg':>10}{'llm/pos':>10}{'blinded':>10}")
    for arg in df["arg"].unique():
        neg = df[(df.sentiment == "negative") & (df.arg == arg)]
        pos = df[(df.sentiment == "positive") & (df.arg == arg)]
        print(f"  {arg:<5}{neg.human.iloc[0]:>10.2f}{pos.human.iloc[0]:>10.2f}"
              f"{neg.llm.iloc[0]:>10.2f}{pos.llm.iloc[0]:>10.2f}{neg.unknown.iloc[0]:>10.2f}")


def main() -> None:
    for name, csv in CORTEGES.items():
        path = DATA / csv
        if path.exists():
            show(name, path)
    print("\nFor cortege (a10, a8, a6), a6 is sent 0.74-1.0 of the time across every setting — the")
    print("'flagship' con argument dominates argument transmission as well as opinion construction.")


if __name__ == "__main__":
    main()

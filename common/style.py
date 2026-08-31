"""Shared figure styling, so every plot in the repository matches."""
FS_LABEL = 15      # axis labels
FS_LEGEND = 15     # legend
FS_TICK = 13       # tick labels and panel titles
FS_TEXT = 7        # in-plot annotations
MS = 7             # marker size
LW = 1             # line width

# per-argument colours, pro first then con
ARG_COLORS = ["#d62728", "#ff7f0e", "#e377c2", "#bcbd22", "#ff9896",
              "#1f77b4", "#17becf", "#9467bd", "#8c564b", "#2ca02c"]

# group colours used by the simulation figures
GROUP_1, GROUP_2 = "#d62728", "#1f77b4"
CLASSICAL_1, CLASSICAL_2 = "#ff7f0e", "#9467bd"

# per-setting colours used by the sending and receiving figures
SETTING = {"human": "#e8000b", "llm": "#1f77d0", "unknown": "#2ca02c"}

TOPOLOGY = {0: "Topology 1 (dense clusters)", 1: "Topology 2 (Erdos-Renyi)"}
PROTOCOL = {"insert-to-end": "insert-to-the-end",
            "insert-to-the-beginning": "insert-to-the-beginning"}

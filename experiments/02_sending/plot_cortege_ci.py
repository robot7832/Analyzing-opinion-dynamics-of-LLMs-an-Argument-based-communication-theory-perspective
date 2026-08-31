# Regenerated for the anonymous artifact repository. Paths are relative to the repository
# root; the API key is read from $OPENROUTER_API_KEY and appears nowhere in the sources.
"""
§7 — the same three arguments before and after permuting the sender's cortege, drawn with the
the maintainer's `plot_argument_confidence_intervals` from `conversational_part1/ArgumentSending.ipynb`.

    (a1, a3, a5)  ->  (a5, a3, a1)

The function is reused as written — its palette, its vertical offsets, its label-placement rule, its
`legend_items`, its printed summary. Only the meaning of `sentiment` is redefined at the data level:

    sentiment = 'positive'  ->  the ORIGINAL cortege order
    sentiment = 'negative'  ->  the PERMUTED order

which is what makes its own `method_line_styles` produce solid for the original order and dashed for
the permuted one, with no change to the drawing code. Variant A: the palette keeps its light/dark
split, so the order is encoded twice over — dark + dashed for the permuted order, light + solid for
the original.

Three changes were unavoidable, all marked `# [patch]` at the point of change:

1. `color_scheme` and `method_vertical_offsets` had no `('negative', 'unknown')` entry, and the
   drawing loop carried an explicit `continue` for that combination. Under the remapping that
   combination is the *permuted blinded* series, so without the entries and without dropping the
   `continue` a third of the figure is silently missing. The added colour is Material dark green, to
   match the dark/light logic of the existing five.
2. Panels are the partner's valence, so the group filter needs one more condition; the original
   plotted every row of the group into one panel.
3. The interval is built from the true trial count when the loader supplies one. The original
   hardcodes `sample_size = 50`, which is right for the human and llm settings but not for the
   blinded one (n = 20), where it makes the intervals ~1.6x too narrow. Set
   `USE_TRUE_COUNTS = False` to get the original behaviour back.

    /opt/anaconda3/bin/python fig_s7_permutation.py                # claude-sonnet-4 (calories)
    /opt/anaconda3/bin/python fig_s7_permutation.py --model gpt    # or gemini
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(parents=True, exist_ok=True)

MODELS = {"claude": ("claude_calories", "claude-sonnet-4 (calories)"),
          "gpt": ("gpt_calories", "gpt-4.1 (calories)"),
          "gemini": ("gemini_calories", "gemini-2.5-flash-lite (calories)")}

# Configuration for confidence intervals
sample_size = 50  # Assumed sample size for proportion calculations
confidence_level = 0.05  # 95% confidence level
USE_TRUE_COUNTS = True    # [patch 3] use the per-cell n from the loader instead of sample_size

fsValue = 15
lsValue = 15
tsValue = 13
textsValue = 7
msValue = 7

lwVal = 1

Labels = [r'$(a_5, a_3, a_1)$', r'$(a_1, a_3, a_5)$']   # [Labels[0] = dashed, Labels[1] = solid]

Titles = ['(a)', '(b)', '(c)', '(d)']

xlabelVal = 'frequency'
ylabelVal = 'argument'

data = None          # set per panel by main(); the function reads it as a global, as in the notebook


def plot_argument_confidence_intervals(sub, argument_tuple, valence, merge_insignificant=True):
    """
    Creates horizontal confidence interval plot for a specific argument tuple.

    Parameters:
    -----------
    argument_tuple : tuple
        Tuple of 3 arguments to analyze (e.g., ('a_1', 'a_3', 'a_5'))
    valence : str
        [patch 2] 'con' / 'pro' — the valence of the argument the partner holds, i.e. the panel.
    merge_insignificant : bool
        If True, merges overlapping intervals that are not significantly different
        If False, shows all intervals separately with method-specific colors
    """

    # Filter data for the specified argument tuple
    group_str = str(argument_tuple)
    group_data = data[(data['group'].astype(str) == group_str)
                      & (data['opp_valence'] == valence)]          # [patch 2]

    # If no data found for the tuple, print message and return
    if group_data.empty:
        print(f"No data found for argument tuple: {argument_tuple}")
        return

    def calculate_confidence_interval(proportion_value, trial_count, method='wilson'):
        """
        Calculates confidence interval for a proportion.
        """
        if pd.isna(proportion_value) or proportion_value < 0 or proportion_value > 1:
            return (np.nan, np.nan)

        n = trial_count if (USE_TRUE_COUNTS and trial_count) else sample_size   # [patch 3]

        # Convert proportion to count (number of successes)
        success_count = int(round(proportion_value * n))

        # Calculate confidence interval using specified method
        return proportion_confint(success_count, n,
                                  alpha=confidence_level, method=method)

    def format_with_interval(value, interval, decimals=2):
        """Форматирует значение с доверительным интервалом в квадратных скобках."""
        if any(np.isnan(conf) for conf in interval):
            return f"{value:.{decimals}f} [N/A]"

        lower, upper = interval
        return f"{value:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"

    def check_interval_overlap(interval1, interval2, overlap_threshold=0.05):
        """Checks if two confidence intervals overlap significantly."""
        lower1, upper1 = interval1
        lower2, upper2 = interval2

        overlap_start = max(lower1, lower2)
        overlap_end = min(upper1, upper2)

        if overlap_start <= overlap_end:
            overlap_length = overlap_end - overlap_start
            length1 = upper1 - lower1
            length2 = upper2 - lower2

            if length1 > 0 and length2 > 0:
                average_length = (length1 + length2) / 2
                return (overlap_length / average_length) > overlap_threshold

        return False

    # Color configuration based on merge setting
    if merge_insignificant:
        # When merging: use sentiment colors only
        color_negative = 'black'
        color_positive = 'gray'
        method_line_styles = {
            'human': '-',     # Solid line for human
            'llm': '--',      # Dashed line for LLM
            'unknown': ':'    # Dotted line for unknown
        }
    else:
        # When not merging: use distinct colors for each (sentiment, method) combination
        color_scheme = {
            ('negative', 'human'): '#D32F2F',    # Dark red
            ('negative', 'llm'): '#1976D2',      # Dark blue
            ('negative', 'unknown'): '#388E3C',  # [patch 1] Dark green
            ('positive', 'human'): '#FF5252',    # Light red
            ('positive', 'llm'): '#64B5F6',      # Light blue
            ('positive', 'unknown'): '#81C784'   # Light green
        }
        # All use solid lines when colors are distinct
        method_line_styles = {
            ('negative', 'human'): '--',    # Dark red
            ('negative', 'llm'): '--',      # Dark blue
            ('negative', 'unknown'): '--',  # [patch 1] Dark green
            ('positive', 'human'): '-',    # Light red
            ('positive', 'llm'): '-',      # Light blue
            ('positive', 'unknown'): '-'   # Light green
        }

    # Data sources to include in the plot
    data_sources = ['human', 'llm', 'unknown']

    # Y-axis positions for each argument
    y_position_mapping = {arg: i+1 for i, arg in enumerate(argument_tuple)}

    # Vertical offsets for different methods (to avoid complete overlap)
    method_vertical_offsets = {
        ('negative', 'human'): -0.3,    # Dark red
        ('negative', 'llm'): -0.17,      # Dark blue
        ('negative', 'unknown'): -0.43,  # [patch 1] Dark green
        ('positive', 'human'): 0.7,    # Light red
        ('positive', 'llm'): 0.17,      # Light blue
        ('positive', 'unknown'): 0.3   # Light green
    }

    # Store intervals for potential merging
    negative_intervals_by_argument = {arg: [] for arg in argument_tuple}
    positive_intervals_by_argument = {arg: [] for arg in argument_tuple}

    # Словари для хранения отформатированных строк с интервалами
    formatted_data_negative = {arg: {source: '' for source in data_sources}
                               for arg in argument_tuple}
    formatted_data_positive = {arg: {source: '' for source in data_sources}
                               for arg in argument_tuple}

    # Process each row in the filtered data
    for _, row in group_data.iterrows():
        current_argument = row['argument']
        current_sentiment = row['sentiment']
        base_y_position = y_position_mapping[current_argument]

        # Process each data source (human, llm, unknown)
        for source in data_sources:
            proportion_value = row[source]
            confidence_interval = calculate_confidence_interval(
                proportion_value, row.get(f'n_{source}'), method='wilson')

            # Сохраняем отформатированную строку
            formatted_str = format_with_interval(proportion_value, confidence_interval)
            if current_sentiment == 'negative':
                formatted_data_negative[current_argument][source] = formatted_str
            else:
                formatted_data_positive[current_argument][source] = formatted_str

            # Skip if confidence interval calculation failed
            if any(np.isnan(conf) for conf in confidence_interval):
                continue

            # Check if we should merge this interval with existing ones
            should_merge_current = False

            if merge_insignificant:
                existing_intervals = (negative_intervals_by_argument[current_argument]
                                      if current_sentiment == 'negative'
                                      else positive_intervals_by_argument[current_argument])

                for existing_ci, existing_source in existing_intervals:
                    if check_interval_overlap(confidence_interval, existing_ci):
                        should_merge_current = True
                        break

            # If not merging, or if we're showing all intervals
            if not should_merge_current or not merge_insignificant:
                # Store interval for potential future merging
                if current_sentiment == 'negative':
                    negative_intervals_by_argument[current_argument].append(
                        (confidence_interval, source))
                else:
                    positive_intervals_by_argument[current_argument].append(
                        (confidence_interval, source))

                # [patch 1] the original dropped ('negative', 'unknown') here; under the remapping
                # that is the permuted blinded series, so the skip is gone.

                # Calculate final y-position with offset
                vertical_offset = method_vertical_offsets[(current_sentiment, source)]
                final_y_position = base_y_position + vertical_offset * 0.3

                # Get appropriate color and style
                if merge_insignificant:
                    line_color = color_negative if current_sentiment == 'negative' else color_positive
                    line_style = method_line_styles[source]
                else:
                    line_color = color_scheme[(current_sentiment, source)]
                    line_style = method_line_styles[(current_sentiment, source)]

                # Draw the confidence interval as a horizontal line
                sub.hlines(y=final_y_position,
                           xmin=confidence_interval[0],
                           xmax=confidence_interval[1],
                           color=line_color,
                           linestyle=line_style,
                           linewidth=lwVal,
                           alpha=0.8)

                # Add point estimate marker
                sub.plot(proportion_value, final_y_position, 'o',
                         color=line_color, markersize=msValue)

                # Add method label near the interval
                label_position = (confidence_interval[1] + 0.02
                                  if confidence_interval[1] < 0.9
                                  else confidence_interval[0] - 0.05)
                sub.text(label_position, final_y_position, source,
                         fontsize=textsValue, color=line_color,
                         verticalalignment='center',
                         horizontalalignment='left' if confidence_interval[1] < 0.9 else 'right')

    # Configure plot appearance
    sub.set_yticks(list(y_position_mapping.values()))
    sub.yaxis.set_tick_params(labelsize=tsValue)
    sub.xaxis.set_tick_params(labelsize=tsValue)
    sub.set_yticklabels(list(y_position_mapping.keys()))
    sub.set_xlabel(xlabelVal, fontsize=fsValue)
    sub.set_ylabel(ylabelVal, fontsize=fsValue)

    # Add grid for better readability
    sub.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis limits
    sub.set_xlim(0, 1.0)

    # Create custom legend based on merge setting
    from matplotlib.lines import Line2D
    if merge_insignificant:
        legend_items = [
            Line2D([0], [0], color='black', lw=2, label=Labels[0]),
            Line2D([0], [0], color='gray', lw=2, label=Labels[1])
        ]
    else:
        legend_items = [
            Line2D([0], [0], color='k', lw=lwVal, ls='--', label=Labels[0]),
            Line2D([0], [0], color='gray', lw=lwVal, label=Labels[1]),
        ]

    # Print data summary with confidence intervals
    print(f"Data Summary for Group: {argument_tuple} — partner holds a {valence} argument")
    print("=" * 70)

    print(f"\npermuted cortege {Labels[0]} (with 95% confidence intervals):")
    print("-" * 70)
    neg_rows = []
    for arg in argument_tuple:
        row = {'argument': arg}
        for source in data_sources:
            row[source] = formatted_data_negative[arg][source]
        neg_rows.append(row)
    print(pd.DataFrame(neg_rows)[['argument', 'human', 'llm', 'unknown']]
          .transpose().to_string(index=True))

    print(f"\noriginal cortege {Labels[1]} (with 95% confidence intervals):")
    print("-" * 70)
    pos_rows = []
    for arg in argument_tuple:
        row = {'argument': arg}
        for source in data_sources:
            row[source] = formatted_data_positive[arg][source]
        pos_rows.append(row)
    print(pd.DataFrame(pos_rows)[['argument', 'human', 'llm', 'unknown']]
          .transpose().to_string(index=True))

    print(f"\nMerging Option: {'ENABLED' if merge_insignificant else 'DISABLED'}")
    print(f"Sample size: {'per-cell n from the data' if USE_TRUE_COUNTS else f'n = {sample_size}'}")
    print(f"Confidence level: {(1-confidence_level)*100:.0f}%")
    print("CI calculation method: Wilson\n")

    return legend_items


def build_data(tag, cortege_original, cortege_permuted):
    """Assemble the notebook's `data` frame from the raw sending decisions.

    Columns: group / argument / sentiment / opp_valence / human / llm / unknown (+ n_* counts).
    `sentiment` carries the cortege ORDER: 'positive' = original, 'negative' = permuted.
    """
    frames = {'positive': (pd.read_csv(HERE / "data" / tag / "sending.csv"), cortege_original),
              'negative': (pd.read_csv(HERE / "data" / tag / "sending_permuted.csv"),
                           cortege_permuted)}
    args_underscored = tuple(f"a_{a[1:]}" for a in
                             (x.strip() for x in cortege_original.strip("()").split(",")))
    rows = []
    for sentiment, (frame, cortege_name) in frames.items():
        sub = frame[frame.cortege == cortege_name]
        for valence in ('con', 'pro'):
            for arg_u in args_underscored:
                arg_raw = 'a' + arg_u[2:]
                row = {'group': str(args_underscored), 'argument': arg_u,
                       'sentiment': sentiment, 'opp_valence': valence}
                for source in ('human', 'llm', 'unknown'):
                    ss = sub[sub.setting == source]
                    if source != 'unknown':          # the blinded setting has no partner valence
                        ss = ss[ss.opp_valence == valence]
                    row[source] = (ss.chosen == arg_raw).mean() if len(ss) else np.nan
                    row[f'n_{source}'] = len(ss)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    global data
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude", choices=list(MODELS))
    a = ap.parse_args()
    tag, model_name = MODELS[a.model]

    data = build_data(tag, '(a1,a3,a5)', '(a5,a3,a1)')

    nrows = 1
    ncols = 2

    fig = plt.figure(figsize=(ncols*5, nrows*5), constrained_layout=True)

    counter = 1

    ############################################################################

    sub = fig.add_subplot(nrows, ncols, counter)

    argument_tuple = ('a_1', 'a_3', 'a_5')

    sub.set_title(f"{Titles[counter-1]} negative-valence setting", fontsize=tsValue)

    legend_items = plot_argument_confidence_intervals(sub, argument_tuple, 'con',
                                                      merge_insignificant=False)

    sub.legend(handles=legend_items,
               loc='upper right',
               fontsize=lsValue)

    counter = counter + 1

    ############################################################################

    sub = fig.add_subplot(nrows, ncols, counter)

    sub.set_title(f"{Titles[counter-1]} positive-valence setting", fontsize=tsValue)

    plot_argument_confidence_intervals(sub, argument_tuple, 'pro', merge_insignificant=False)

    counter = counter + 1

    ############################################################################

    # no suptitle: the model and topic belong in the manuscript caption, not on the artwork

    out = FIG / f"s7_permutation_{a.model}_calories"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out.name}.png (+ .pdf)")


if __name__ == "__main__":
    main()

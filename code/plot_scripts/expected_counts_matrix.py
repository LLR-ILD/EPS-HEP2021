#!/usr/bin/env python
import sys
import typing
from pathlib import Path

import alldecays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

assert len(sys.argv) == 4
save_to = sys.argv[1]
fancy_names_path = Path(sys.argv[2])
with fancy_names_path.open() as f:
    fancy_names = yaml.safe_load(f)
file = Path(sys.argv[3])
assert file.is_file()
assert file.suffix == ".csv"
assert file.stem.startswith("counts_")
data = pd.read_csv(file, index_col=0)
is_higgs_decay = data.pop("is_higgs_decay")
if "bkg" not in save_to:
    data = data[is_higgs_decay]
failed_presel_counts = data.pop("failed_presel")
data = data.transpose()
as_probability = "probability" in save_to
if as_probability:
    all_process_counts = data.sum() + failed_presel_counts
    data = 100 * data / all_process_counts
sample_name = file.stem.replace("counts_", "")
sample_name = {"e1e1": r"$Z\to e^+e^-$", "e2e2": r"$Z\to \mu^+\mu^-$"}[sample_name]


def set_numbers(ax, matrix, omit_zero=True):
    np_matrix = np.array(matrix)  # To make it work also for pd.DataFrame.
    for text_y, row in enumerate(np_matrix):
        for text_x, val in enumerate(row):
            if omit_zero and val == 0:
                continue

            if val > 0.3 * np_matrix.max():
                color = "black"
            else:
                color = "white"

            ax.text(
                text_x,
                text_y,
                alldecays.plotting.channel.matrix_plots._my_format(val),
                ha="center",
                va="center",
                color=color,
            )


fig, ax = plt.subplots(figsize=(10, 15))
ax.imshow(data)
set_numbers(ax, data)

label_size = 16
ild_tag_size = label_size + 2
x_tick_size = label_size - 2
ax.text(
    1,
    1.005,
    "ILD preliminary",
    ha="right",
    va="bottom",
    transform=ax.transAxes,
    color="gray",
    weight="bold",
    alpha=1,
    fontsize=ild_tag_size,
)

if as_probability:
    ax.set_title(
        "Matrix entries\nP(Class|BR) [%]",
        fontsize=label_size,
        loc="left",
    )
else:
    n_signal = data[data.columns[is_higgs_decay]].sum().sum()
    ax.set_title(
        f"SM event distribution \n{n_signal:.0f} signal events in {sample_name}",
        fontsize=label_size,
        loc="left",
    )

x_tick_labels = [fancy_names.get(process, process) for process in data.columns]
y_tick_labels: typing.List[str] = []  # [br for br in data.index]
ax.set_xticks(np.arange(len(x_tick_labels)))
ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=x_tick_size)
ax.set_yticks(np.arange(len(y_tick_labels)))
ax.set_yticklabels(y_tick_labels, fontsize=label_size)
ax.set_xlabel("BR", fontsize=label_size)
ax.set_ylabel("event class", fontsize=label_size)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(save_to, dpi=300, transparent=True)

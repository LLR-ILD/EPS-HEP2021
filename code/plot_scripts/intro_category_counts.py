#!/usr/bin/env python
import sys
import typing
from pathlib import Path

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
data.pop("failed_presel")
is_higgs_decay = data.pop("is_higgs_decay")
n_counts = data.pop("n_counts")
data = data.mul(n_counts, axis=0)
sample_name = file.stem.replace("counts_", "")
sample_name = {"e1e1": r"$Z\to e^+e^-$", "e2e2": r"$Z\to \mu^+\mu^-$"}[sample_name]


fig, ax = plt.subplots(figsize=(4, 4))
higgs_decay_names = data.index[is_higgs_decay]

y = np.arange(len(data.columns))
left = np.zeros_like(y)
for process_name in higgs_decay_names:
    counts = data.loc[process_name]
    if "signal_composition" in save_to:
        if "w_bkg" in save_to:
            divide_by = data.sum(axis=0)
        else:
            divide_by = data.loc[is_higgs_decay].sum(axis=0)
        counts = counts / divide_by
    label = fancy_names.get(process_name, process_name)
    ax.barh(y, counts, left=left, label=label)
    left = left + counts

data_counts = data.sum(axis=0)
ax.set_xlim((0, 1.05 * max(data_counts)))
if "w_bkg" in save_to and "signal_composition" not in save_to:
    ax.scatter(
        data_counts,
        np.arange(len(data_counts)),
        label="including bkg",
        color="black",
    )

ax.set_xlabel("expected signal counts")
if "signal_composition" in save_to:
    ax.set_xlabel("expected signal composition")
    ax.set_xlim((0, 1))
ax.set_ylabel("event class")
y_tick_labels: typing.List[str] = []  # [br for br in data.index]
ax.set_yticks(np.arange(len(y_tick_labels)))
ax.set_yticklabels(y_tick_labels)
ax.text(0, 1.005, sample_name, ha="left", va="bottom", transform=ax.transAxes)
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
    fontsize=12,
)
ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(save_to, dpi=300, transparent=True)

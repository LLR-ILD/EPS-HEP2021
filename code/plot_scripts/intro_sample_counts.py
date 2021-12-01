#!/usr/bin/env python
import sys
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

save_to = sys.argv[1]
fancy_names_path = Path(sys.argv[2])
with fancy_names_path.open() as f:
    fancy_names = yaml.safe_load(f)
data = {}
is_higgs_decay = {}
for file in map(Path, sys.argv[3:]):
    assert file.is_file()
    assert file.suffix == ".csv"
    assert file.stem.startswith("counts_")

    sample_data = pd.read_csv(file, index_col=0)
    sample_data.pop("failed_presel")
    is_higgs_decay[file.stem.replace("counts_", "")] = sample_data.pop("is_higgs_decay")
    n_counts = sample_data.pop("n_counts")
    sample_data = sample_data.mul(n_counts, axis=0)
    data[file.stem.replace("counts_", "")] = sample_data


fig, ax = plt.subplots(figsize=(4, 4))
sample_counts = pd.DataFrame()
higgs_decay_names: typing.List[str] = []
for channel in data:
    if len(higgs_decay_names) == 0:
        higgs_decay_names = data[channel].index[is_higgs_decay[channel]]
    else:
        assert set(higgs_decay_names) == set(
            data[channel].index[is_higgs_decay[channel]]
        )
    sample_counts[channel] = data[channel].sum(axis=1)

y = np.arange(len(data))
left = np.zeros_like(y)
for process_name in sample_counts.index:
    if process_name in higgs_decay_names:
        alpha = 1.0
    else:
        alpha = 0.6
        continue  # Use either this or full_bkg
    counts = sample_counts.loc[process_name]
    label = fancy_names.get(process_name, process_name)
    ax.barh(y, counts, left=left, label=label, alpha=alpha)
    left = left + counts


def full_bkg():
    data_counts = sample_counts.sum(axis=0)
    ax.scatter(
        data_counts,
        np.arange(len(data_counts)),
        label="including bkg",
        color="black",
    )
    ax.set_xlim((0, 1.05 * max(data_counts)))


full_bkg()

ax.legend(loc="center right")
ax.set_xlabel("expected signal counts")
ax.set_yticks(y)
ax.set_yticklabels(
    [
        {"e1e1": r"$Z\to e^+e^-$", "e2e2": r"$Z\to \mu^+\mu^-$"}[c]
        for c in sample_counts.columns
    ]
)
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
fig.tight_layout()
fig.savefig(save_to, dpi=300, transparent=True)

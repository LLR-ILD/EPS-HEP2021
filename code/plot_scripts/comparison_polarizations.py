#!/usr/bin/env python
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

assert len(sys.argv) == 4
save_to = Path(sys.argv[1])
fancy_names_path = Path(sys.argv[2])
with fancy_names_path.open() as f:
    fancy_names = yaml.safe_load(f)
error = pd.read_csv(sys.argv[3], index_col=0)


fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(len(error))
ax.set_ylabel("Relative uncertainty change")
ax.set_xticks(x)
ax.set_xticklabels([fancy_names.get(n, n) for n in error.index], rotation=90)

default = error.pop("default")
error = error.div(default, axis=0)
ax.set_ylim(error.min().min() - 0.05, error.max().max() + 0.05)
ax.scatter(x, np.ones_like(x), marker="*", color="C1", label="Standard Model")

for scenario in error.columns:
    color = dict(
        eLpR="blue",
        eRpL="tab:red",
        unpolarized="gray",
    ).get(scenario, None)
    ax.scatter(x, error[scenario], marker="_", color=color, label=scenario)
    ax.plot(x, error[scenario], color=color, alpha=1)


def add_watermark(ax):
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


add_watermark(ax)
ax.legend(
    title="Beam polarization", loc="upper left", bbox_to_anchor=(1, 1.05), fontsize=11
)
fig.savefig(save_to, dpi=300, transparent=True, bbox_inches="tight")

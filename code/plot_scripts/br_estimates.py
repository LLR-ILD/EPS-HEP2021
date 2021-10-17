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
data = pd.read_csv(file, index_col=0)

fig, axs = plt.subplots(
    figsize=(4, 6), nrows=3, sharex=True, gridspec_kw={"height_ratios": [6, 2, 3]}
)
x = np.arange(len(data))
legend_items: typing.Any = [None] * 3
legend_items[2] = axs[0].bar(
    x, 100 * data.starting_values, alpha=0.75, label="Fit start BRs", color="C0"
)
kw_expected_fit = dict(
    x=x,
    y=100 * data.fit_values,
    yerr=100 * data.errors,
    xerr=0.3,
    fmt="o",
    label="Expected class counts",
    color="C1",
    alpha=0.8,
)
kw_toy_scatter = dict(
    x=x,
    y=100 * data.toy_values,
    marker="*",
    label="Toy data counts",
    color="C2",
    zorder=10,
)
legend_items[0] = axs[0].errorbar(**kw_expected_fit)
legend_items[1] = axs[0].scatter(**kw_toy_scatter)

if "changed" in save_to:
    legend_items.append(
        axs[0].bar(
            x,
            100 * data.data_values,
            hatch="//",
            fill=False,
            edgecolor="black",
            label="Expected BRs in data",
        )
    )


def add_inset(ax):
    ax_inset = ax.inset_axes([0.4, 0.2, 0.55, 0.3])
    ax_inset.errorbar(**kw_expected_fit)
    ax_inset.scatter(**kw_toy_scatter)
    ax_inset.set_xticks(np.arange(len(x)))
    ax_inset.set_xticklabels([""] * len(x))
    y_max = 0.75
    x_min = 6
    ax_inset.set_xlim((x_min - 0.5, len(x) - 0.5))
    ax_inset.set_ylim((0, y_max))
    ax.indicate_inset_zoom(ax_inset, edgecolor="grey")


add_inset(axs[0])
axs[0].set_ylabel("branching ratio [%]")
axs[0].set_ylim((0, None))
axs[0].legend(
    legend_items,
    [li.get_label() for li in legend_items],
    title="Fit with",
    loc="upper right",
)
axs[0].text(
    1,
    1.005,
    "ILD preliminary",
    ha="right",
    va="bottom",
    transform=axs[0].transAxes,
    color="gray",
    weight="bold",
    alpha=1,
    fontsize=12,
)

pull = (data.toy_values - data.data_values) / data.errors
axs[1].scatter(
    x,
    pull,
    marker="*",
    color="C2",
    zorder=10,
)
axs[1].set_ylabel("pull")
axs[1].set_ylim((-1.2 * max(abs(pull)), 1.2 * max(abs(pull))))

axs[2].bar(
    x,
    100 * data.errors,
    width=0.2,
    color="C1",
    zorder=10,
)
axs[2].set_ylabel("67% CL [%]")
axs[2].set_ylim((0, None))

xticklabels = [fancy_names.get(br, br) for br in data.index]
axs[-1].set_xticks(np.arange(len(xticklabels)))
axs[-1].set_xticklabels(xticklabels, rotation=90)
fig.tight_layout()
plt.subplots_adjust(hspace=0.075)
fig.savefig(save_to, dpi=300, transparent=True)

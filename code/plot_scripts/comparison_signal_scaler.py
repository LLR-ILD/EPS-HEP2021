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

no_bkg_label = "$ \\times 1 $, no bkg"
if no_bkg_label in error.columns:
    no_bkg = error.pop(no_bkg_label)
    kw = dict(color="blue")
    y = no_bkg
    ax.scatter(x, y, marker="_", label=no_bkg_label, **kw)
    ax.plot(x, y, **kw)


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


def do_saving(path):
    add_watermark(ax)
    ax.legend(
        title=r"$\sigma_{\mathrm{ZH}}$ rescaled",
        loc="upper left",
        bbox_to_anchor=(1, 1.05),
    )
    # fig.tight_layout()
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight")


def add_signal_rescaled():
    for i, scenario in enumerate(error.columns):
        kw = dict(
            color="gray",
            alpha=1 - 0.6 * i / len(error.columns),
        )
        y = error[scenario]
        ax.scatter(x, y, marker="_", label=scenario, **kw)
        ax.plot(x, y, **kw)


partial_to = save_to.parent / f"{save_to.stem}_partial{save_to.suffix}"
do_saving(partial_to)
add_signal_rescaled()
do_saving(save_to)
partial_to.touch()  # Ensure that the partial file is newer - needed in Makefile.

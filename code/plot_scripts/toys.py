#!/usr/bin/env python
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

assert len(sys.argv) == 5
save_to = Path(sys.argv[1])
assert save_to.stem.endswith("_bb")
fancy_names_path = Path(sys.argv[2])
with fancy_names_path.open() as f:
    fancy_names = yaml.safe_load(f)
file_toys = Path(sys.argv[3])
assert file_toys.is_file()
assert file_toys.suffix == ".csv"
data_toys = pd.read_csv(file_toys)
file_expected = Path(sys.argv[4])
assert file_expected.is_file()
assert file_expected.suffix == ".csv"
data_expected = pd.read_csv(file_expected, index_col=0)

figsize = (4, 4)
bins = 20


def redo_label(label):
    number = label.split(" ")[0]
    if not number.isnumeric():
        return label
    new_number = f"{100 * float(number):.2f}%"
    return label.replace(number, new_number)


def format_x_percent(x, y):
    if x < 0.1:
        return f"{100 * x:.2f}"
    else:
        return f"{100 * x:.1f}"


def add_not_accurate_indication(is_accurate_fit, values, edges, ax):
    is_not_accurate = np.logical_not(is_accurate_fit)
    if sum(is_not_accurate) > 0:
        counts, _ = np.histogram(values[is_not_accurate], edges, density=True)
        ax.bar(
            (edges[:-1] + edges[1:]) / 2,
            counts * sum(is_not_accurate) / len(is_not_accurate),
            (edges[:-1] - edges[1:]),
            hatch="///",
            color="gray",
            alpha=0.5,
            label="fits tagged 'not accurate'",
        )


def add_ild_watermark(ax):
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


def add_starting_value_line(ax, expected):
    ax.axvline(
        expected.starting_values,
        color="grey",
        linestyle=":",
        linewidth=2,
        zorder=3,
        label=f"{100 * expected.starting_values:.2f}% SM BR",
    )


def add_gaussian_uncertainty(ax, expected, edges):
    def _gauss(x, mu, sigma):
        """1D Gaussian distribution"""
        return (2 * np.pi * sigma ** 2) ** -0.5 * np.exp(
            -0.5 * (x - mu) ** 2 / sigma ** 2
        )

    exp_br_i = expected.fit_values
    exp_err_i = expected.errors
    ax.axvline(exp_br_i, color="black", label=f"{100 * exp_br_i:.2f}% EECF Minimum")
    x = np.linspace(edges[0], edges[-1], 1000)
    ax.plot(
        x,
        _gauss(x, exp_br_i, exp_err_i),
        color="C0",
        label=f"{100*exp_err_i:0.2f}% σ from EECF",
    )


brs = list(data_expected.index)
# bb should be last figure created, given our Makefile construction for changed BRs.
brs.insert(0, brs.pop(brs.index("bb")))
for br in brs:
    toy_values = data_toys[br]
    expected = data_expected.loc[br]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(fancy_names.get(br, br), loc="left")

    n, edges, _ = ax.hist(
        toy_values,
        bins,
        label=f"Minima from {len(toy_values)} fits\non toy counts\n(Multinomial draws)",
        color="C1",
        density=True,
    )
    add_starting_value_line(ax, expected)
    add_gaussian_uncertainty(ax, expected, edges)
    add_not_accurate_indication(data_toys.is_accurate_fit, toy_values, edges, ax)
    add_ild_watermark(ax)
    if br == "az" and "default" in save_to.stem:
        # Only draw the legend in a specific case.
        ax.legend(title="EECF: Minuit fit on the\nexpected event counts")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_x_percent))
    ax.set_xlabel("BR [%]")
    fig.tight_layout()
    br_save_to = save_to.parent / f"{save_to.stem[:-3]}_{br}{save_to.suffix}"
    fig.savefig(br_save_to, dpi=300, transparent=True)

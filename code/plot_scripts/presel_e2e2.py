#!/usr/bin/env python
import re
import sys
from pathlib import Path

import higgstables
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_to = Path(sys.argv[1])
assert save_to.stem.split("_")[-1].isnumeric()
step_info = pd.read_csv(sys.argv[2], index_col=0)
steps = {}
for file in map(Path, sys.argv[3:]):
    assert file.is_file()
    assert file.suffix == ".csv"
    assert file.stem.startswith("presel_e2e2_")
    steps[file.stem.replace("presel_e2e2_", "")] = pd.read_csv(file)
signal_process = "Pe2e2h"
assert len(step_info) == len(steps)


def get_latex(string):
    """Quick fix for nicer figure texts."""
    for normal, latex in {
        "m_z": "$M_Z$",
        "m_recoil": r"$M_{\mathrm{recoil}}$",
        "abs(cos_theta_z)": "$| \\mathrm{cos} \\theta_Z |$",
        "abs(cos_theta_miss)": "$| \\mathrm{cos} \\theta_{\\mathrm{miss}} |$",
        "abs(cos_theta_z - cos_theta_miss)": "$| \\mathrm{cos} \\theta_Z - \\mathrm{cos} \\theta_{\\mathrm{miss}} |$",
        "cos_theta_z": "$\\mathrm{cos} \\theta_Z$",
        "cos_theta_miss": "$\\mathrm{cos} \\theta_{\\mathrm{miss}}$",
    }.items():
        if normal in string:
            string = string.replace(normal, latex)
    return string


def get_name_of_variable(step_name):
    step_vars = set(higgstables.config.util.get_variables_from_expression(step_name))
    if len(step_vars) == 1:
        var_name = step_vars.pop()
    else:
        pattern = re.compile(">|<|>=|<=|==")
        var_name = pattern.split(step_name)[0]
    return get_latex(var_name)


def get_process_group_order(df):
    group_order = list(df.sum().sort_values(ascending=False).index)
    # Move signal process to front.
    group_order.insert(0, group_order.pop(group_order.index(signal_process)))
    return group_order


nrows = 1 + (len(steps) - 1) // 2
fig, axs = plt.subplots(ncols=2, nrows=nrows, figsize=(10, 4 * nrows))
axs = axs.flatten()
group_order = get_process_group_order(steps["step1"])
for i in range(1, len(steps) + 1):
    ax = axs[i - 1]
    step_data = steps[f"step{i}"]
    step_name, x_min, x_max, n_bins = step_info.loc[f"step_{i}"]
    binning = np.linspace(x_min, x_max, n_bins + 1)

    x = (binning[1:] + binning[:-1]) / 2
    kw = dict(
        width=binning[1:] - binning[:-1],
        alpha=1,
    )
    bottom = np.zeros_like(step_data[signal_process])
    for group in group_order:
        ax.bar(x, step_data[group], bottom=bottom, label=group, **kw)
        bottom += step_data[group]
        kw["alpha"] = 0.5  # First group was signal, now bkg groups.

    # Add horizontal lines
    for cut_value in re.findall(r"[-+]?\d*\.\d+|\d+", step_name):
        ax.axvline(float(cut_value), ls="--", color="black")
        if step_name.startswith("abs(") and min(binning) < 0:
            ax.axvline(-float(cut_value), ls="--", color="black")

    # ax.set_xlabel(get_name_of_variable(step_name), fontsize=18)
    t = ax.text(
        0.5,
        0.5,
        get_name_of_variable(step_name),
        ha="center",
        va="center",
        transform=ax.transAxes,
        # color="gray",
        weight="bold",
        # alpha=1,
        fontsize=25,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round"))
    ax.set_label("weighted bin counts")
    ax.set_title(f"step {i}: {get_latex(step_name)}")

axs[0].legend(bbox_to_anchor=(0, 1), loc="upper left")
t = axs[0].text(
    0.995,
    0.995,
    "ILD preliminary",
    ha="right",
    va="top",
    transform=axs[0].transAxes,
    color="gray",
    weight="bold",
    alpha=1,
    fontsize=14,
    zorder=99,
)
t.set_bbox(
    dict(facecolor="white", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.02")
)
for ax in axs:
    ax.set_visible(False)
for i in range(1, len(steps) + 1):
    new_stem_it = "_".join(save_to.stem.split("_")[:-1]) + f"_{i}"
    it_saved = save_to.parent / (new_stem_it + save_to.suffix)
    axs[i - 1].set_visible(True)
    fig.savefig(it_saved, dpi=300, transparent=True)

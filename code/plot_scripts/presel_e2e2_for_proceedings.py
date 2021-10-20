#!/usr/bin/env python
import re
import sys
from pathlib import Path

import higgstables
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_to = Path(sys.argv[1])
eff_pur_df = pd.read_csv(sys.argv[2])
step_info = pd.read_csv(sys.argv[3], index_col=0)
steps = {}
for file in map(Path, sys.argv[4:]):
    assert file.is_file()
    assert file.suffix == ".csv"
    assert file.stem.startswith("presel_e2e2_")
    steps[file.stem.replace("presel_e2e2_", "")] = pd.read_csv(file)
signal_process = "Pe2e2h"
assert len(step_info) == len(steps)


def add_eff_pur(ax):
    eff_pur_df["eff_pur"] = eff_pur_df.efficiency * eff_pur_df.purity
    x = np.arange(len(eff_pur_df))
    ax.plot(x[: i + 1], eff_pur_df.efficiency[: i + 1], label="efficiency", marker="o")
    ax.plot(x[: i + 1], eff_pur_df.purity[: i + 1], label="purity", marker="d")
    ax.plot(x[: i + 1], eff_pur_df.eff_pur[: i + 1], label="eff * pur", marker="*")
    ax.set_xticks(x)
    ax.set_xlabel("step")
    ax.set_xlim((min(x) - 0.5, max(x) + 0.5))
    ax.set_ylim((0, 1))
    ax.legend(bbox_to_anchor=(1, 0.9), loc="upper right")


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


n_eff_axes = 1
ncols = 1 + (len(steps) - 1 + n_eff_axes) // 2
fig, axs = plt.subplots(ncols=ncols, nrows=2, figsize=(ncols * 5, 4 * 2))
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

add_eff_pur(axs[len(steps)])
for ax in axs[len(steps) + 1 :]:
    ax.set_visible(False)
fig.savefig(save_to, dpi=300, transparent=True)

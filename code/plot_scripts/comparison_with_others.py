#!/usr/bin/env python
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

assert len(sys.argv) == 5
save_to = Path(sys.argv[1])
fancy_names_path = Path(sys.argv[2])
with fancy_names_path.open() as f:
    fancy_names = yaml.safe_load(f)
luminosity_path = Path(sys.argv[3])
with luminosity_path.open() as f:
    my_luminosity = float(f.read())
file_fit = Path(sys.argv[4])
assert file_fit.is_file()
assert file_fit.suffix == ".csv"
data_fit = pd.read_csv(file_fit, index_col=0)

# -----------------------------------------------------------------------------
#
# This block collects some global fits from other groups.
#
global_fit_ilc_250 = {  # Table 1 in J.Tian & K. Fujii. https://www.sciencedirect.com/science/article/pii/S2405601415006161
    "cc": 0.068,
    "bb": 0.053,
    "e3e3": 0.057,
    "gg": 0.064,
    "aa": 0.18,
    "zz": 0.08,
    "ww": 0.048,
    # "ΓH":       0.11,
    "inv": 0.0095,
}
for k, v in global_fit_ilc_250.items():
    # Scale errors to the H20 ILC250 scenario that I am considering.
    global_fit_ilc_250[k] = v * (250 / my_luminosity) ** 0.5

# SFitter: 3000 ifb of 14 TeV LHC,
#           250 ifb of 250 GeV ILC.
# Table I in SFitter 2013: https://inspirehep.net/literature/1209590
sfitter_hl_lhc = {
    "bb": 0.17,
    "e3e3": 0.09,
    "gg": 0.105,
    "aa": 0.08,
    "zz": 0.08,
    "ww": 0.075,
}
sfitter_hl_lhc_improved = {
    "bb": 0.145,
    "e3e3": 0.07,
    "gg": 0.095,
    "aa": 0.06,
    "zz": 0.07,
    "ww": 0.06,
}
sfitter_ilc = {
    "bb": 0.145,
    "cc": 0.095,
    "e3e3": 0.08,
    "gg": 0.08,
    "aa": 0.155,
    "zz": 0.015,
    "ww": 0.055,
}
sfitter_ilc_scaled = {}
for k, v in sfitter_ilc.items():
    # Scale errors to the H20 ILC250 scenario that I am considering.
    sfitter_ilc_scaled[k] = v * (250 / my_luminosity) ** 0.5
sfitter_lhc_ilc = {
    "bb": 0.045,
    "cc": 0.06,
    "e3e3": 0.05,
    "gg": 0.06,
    "aa": 0.06,
    "zz": 0.0095,
    "ww": 0.04,
}

peskin_ilc_zh = {  # https://arxiv.org/abs/1207.2516
    # "zz":     0.19,  # σ(ZH)BR(ZZ)
    "zz": 0.025,  # σ(ZH)
    # "bb":     0.105,  # σ(WW)BR(bb)
    "bb": 0.011,  # σ(ZH)BR(bb)
    "cc": 0.074,  # σ(ZH)BR(cc)
    "e3e3": 0.042,  # σ(ZH)BR(ττ)
    "gg": 0.06,  # σ(ZH)BR(bb)
    "aa": 0.38,  # σ(ZH)BR(γγ)
    "ww": 0.064,  # σ(ZH)BR(WW)
    "inv": 0.005,  # σ(ZH)BR(inv.)
}
for k, v in peskin_ilc_zh.items():
    # Scale errors to the H20 ILC250 scenario that I am considering.
    peskin_ilc_zh[k] = v * (250 / my_luminosity) ** 0.5


coupling_projections = {}
coupling_projections["ILC 250 global coupling fit [1]"] = global_fit_ilc_250
# coupling_projections["SFitter HL-LHC"] = sfitter_hl_lhc
coupling_projections["SFitter HL-LHC improved [2]"] = sfitter_hl_lhc_improved
# coupling_projections["SFitter ILC250 250 ifb"] = sfitter_ilc
# coupling_projections["SFitter ILC250 scaled Lumi"] = sfitter_ilc_scaled
# coupling_projections["SFitter LHC+(ILC250 250 ifb)"] = sfitter_lhc_ilc
# coupling_projections["Peskin 2012 through σ(ZH)"] = peskin_ilc_zh


# -----------------------------------------------------------------------------
#
# Now the plotting.
#
def add_inset(ax):
    ax_inset = ax.inset_axes([0.35, 0.2, 0.55, 0.35])
    ax_inset.set_xticks(np.arange(len(x)))
    ax_inset.set_xticklabels([""] * len(x))
    y_max = 0.5
    x_min = 0
    ax_inset.set_xlim((x_min - 0.5, len(x) - 0.5))
    ax_inset.set_ylim((0, y_max))
    rectpatch, connects = ax.indicate_inset_zoom(ax_inset, edgecolor="grey")
    return ax_inset


fig, ax = plt.subplots(figsize=(4, 3))
x = np.arange(len(data_fit))
axs = [ax, add_inset(ax)]

for _ax in axs:
    _ax.scatter(
        x,
        100 * data_fit.errors,
        marker="*",
        color="C1",
        label="This fit",
    )
for name, coupling_unc in coupling_projections.items():
    br_error = [None] * len(data_fit)
    for i, br_name in enumerate(data_fit.index):
        if br_name not in coupling_unc:
            continue
        coupling_unc[br_name]
        [coupling_unc.get(br, None) for br in data_fit.index]
        # coupling_projections is citing relative coupling errors.
        # `* 2` to move to BRs. `* br` to move to total errors.
        br_error[i] = 100 * coupling_unc[br_name] * 2 * data_fit.data_values[br_name]
    if "SFitter HL-LHC improved" in name:
        color = "tab:red"
    elif "ILC 250 global coupling fit" in name:
        color = "blue"
    else:
        color = f"C{i+2}"
    for _ax in axs:
        _ax.scatter(
            x,
            br_error,
            marker="_",
            color=color,
            label=name,
        )


xticklabels = [fancy_names.get(br, br) for br in data_fit.index]
ax.set_xticks(np.arange(len(xticklabels)))
ax.set_xticklabels(xticklabels, rotation=90)
ax.set_ylabel("Branching ratio uncertainty [%]")
ax.set_ylim((0, None))


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
ax.legend()
fig.tight_layout()
fig.savefig(save_to, dpi=300, transparent=True)

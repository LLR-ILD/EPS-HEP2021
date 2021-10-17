#!/usr/bin/env python
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

assert len(sys.argv) == 3
save_to = Path(sys.argv[1])
assert save_to.stem.split("_")[-1].isnumeric()
eff_pur_df = pd.read_csv(sys.argv[2])

eff_pur_df["eff_pur"] = eff_pur_df.efficiency * eff_pur_df.purity
x = np.arange(len(eff_pur_df))
for i in x:
    fig, ax = plt.subplots(figsize=(4, 3))
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
    ax.plot(x[: i + 1], eff_pur_df.efficiency[: i + 1], label="efficiency", marker="o")
    ax.plot(x[: i + 1], eff_pur_df.purity[: i + 1], label="purity", marker="d")
    ax.plot(x[: i + 1], eff_pur_df.eff_pur[: i + 1], label="eff * pur", marker="*")
    ax.set_xticks(x)
    ax.set_xlabel("step")
    ax.set_xlim((min(x) - 0.5, max(x) + 0.5))
    ax.set_ylim((0, 1))
    ax.legend(bbox_to_anchor=(1, 0.9), loc="upper right")
    fig.tight_layout()

    new_stem_it = "_".join(save_to.stem.split("_")[:-1]) + f"_{i}"
    it_saved = save_to.parent / (new_stem_it + save_to.suffix)
    fig.savefig(it_saved, dpi=300, transparent=True)

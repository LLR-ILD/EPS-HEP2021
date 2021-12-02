#!/usr/bin/env python
import sys
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

brs = data.index
cov = data[brs].values
corr = (cov / cov.diagonal() ** 0.5).T / cov.diagonal() ** 0.5
corr = corr - np.eye(corr.shape[0])  # We do not want to color the diagonal.

fig, ax = plt.subplots(figsize=(5, 5))
xticklabels = [fancy_names.get(br, br) for br in brs]
yticklabels = xticklabels
ax.set_xticks(np.arange(len(xticklabels)))
ax.set_yticks(np.arange(len(yticklabels)))
ax.set_xticklabels(xticklabels, rotation=90)
ax.set_yticklabels(yticklabels)

for text_y, row in enumerate(corr):
    for text_x, val in enumerate(row):
        if text_x == text_y:
            continue
        color = "black"  # if val > 0.3 * corr.max() else "white"
        ax.text(
            text_x,
            text_y,
            f"{val:.3f}".replace("0.", "."),
            ha="center",
            va="center",
            color=color,
        )
ax.imshow(corr, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
ax.set_title("Correlations", loc="left")
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

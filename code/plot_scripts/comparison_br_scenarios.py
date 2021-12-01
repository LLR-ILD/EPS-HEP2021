#!/usr/bin/env python
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

save_to = sys.argv[1]
fancy_names_path = Path(sys.argv[2])
with fancy_names_path.open() as f:
    fancy_names = yaml.safe_load(f)
data = pd.DataFrame()
error = pd.DataFrame()
for file in map(Path, sys.argv[3:]):
    assert file.is_file()
    assert file.suffix == ".csv"
    assert file.stem.startswith("fit_")

    sample_data = pd.read_csv(file, index_col=0)
    data[file.stem.replace("fit_", "")] = sample_data["data_values"]
    error[file.stem.replace("fit_", "")] = sample_data["errors"]
error = error.loc[~(data == 0).all(axis=1)]
data = data.loc[~(data == 0).all(axis=1)]
data = data.replace(0, 1e-4)


fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(len(data))
ax.scatter(x, np.ones_like(x), marker="*", color="C1", label="Standard Model")
for scenario in data.columns:
    if scenario == "default":
        continue
    e2e2_unc = 100 * error[scenario]["e2e2"]
    label = dict(
        changed_bbww="$\\searrow 15 \\%: H \\to bb$\n$\\nearrow 15\\%: H \\to WW$",
        changed_e2e2="\n".join(
            [
                r"$BR(H \to \mu \mu) = 1 \%$",
                "($\\sigma_{\\mathrm{stat}} ^ {\\mu \\mu} = " + fr"{e2e2_unc:.1f} \%$)",
            ]
        ),
    ).get(scenario, scenario)
    color = dict(changed_bbww="blue", changed_e2e2="tab:red").get(scenario, None)
    y = (error[scenario] / data[scenario]) / (error["default"] / data["default"])
    ax.scatter(x, y, marker="_", color=color, label=label)
    ax.plot(x, y, color=color, alpha=1)
ax.set_ylabel("Relative uncertainty change")
ax.set_xticks(x)
ax.set_xticklabels([fancy_names.get(n, n) for n in data.index], rotation=90)


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
ax.legend(title="Higgs BRs", loc="upper left", bbox_to_anchor=(1, 1.05))
fig.savefig(save_to, dpi=300, transparent=True, bbox_inches="tight")

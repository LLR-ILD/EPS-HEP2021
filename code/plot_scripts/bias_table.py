#!/usr/bin/env python
import sys
from pathlib import Path

import pandas as pd
import yaml

assert len(sys.argv) == 4
save_to = Path(sys.argv[1])
assert save_to.suffix == ".tex"
fancy_names_path = Path(sys.argv[2])
with fancy_names_path.open() as f:
    fancy_names = yaml.safe_load(f)
file = Path(sys.argv[3])
assert file.is_file()
assert file.suffix == ".csv"
data = pd.read_csv(file, index_col=0)

df = pd.DataFrame(
    {
        "SM BR": 100 * data.starting_values,
        # "minimum": 100 * data.values,  # TODO: Put back in for MC bias.
        r"$\sigma_{\mathrm{stat}}$": 100 * data.errors,
    },
)
df.index = [fancy_names.get(br, br) for br in data.index]
df.to_latex(buf=save_to, float_format="%0.2f", escape=False)

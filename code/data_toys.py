#!/usr/bin/env python
import alldecays
import numpy as np
import pandas as pd


def data_toys(data_set, HIGGS_BR, HIGGS_BR_DATA, n_toys=10_000):
    def get_br_values(br_dict):
        assert set(data_set.decay_names) == br_dict.keys()
        brs = np.array([br_dict[k] for k in data_set.decay_names])
        return brs

    data_set.data_brs = get_br_values(HIGGS_BR_DATA)
    data_set.fit_start_brs = get_br_values(HIGGS_BR)
    fit = alldecays.Fit(
        data_set,
        fit_mode="BinomialLeastSquares",
        has_limits=True,
    )
    toys = fit.fill_toys(n_toys=n_toys, rng=np.random.default_rng(seed=1))
    df = pd.DataFrame(toys.physics, columns=fit.fit_mode.parameters)
    df["is_accurate_fit"] = toys.accurate
    return df


if __name__ == "__main__":
    import sys

    from helper_data import get_data_set, read_settings

    output_file, input_dir, s = read_settings(sys.argv)
    data_set = get_data_set(input_dir, **s)
    if "changed" in output_file.stem:
        changed_id = output_file.stem.split("changed_")[-1]
        HIGGS_BR_DATA = s[f"HIGGS_BR_CHANGED_{changed_id}"]
    else:
        HIGGS_BR_DATA = s["HIGGS_BR"]
    df = data_toys(data_set, s["HIGGS_BR"], HIGGS_BR_DATA, n_toys=10_000)
    df.to_csv(str(output_file), index=False)

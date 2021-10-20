#!/usr/bin/env python
import alldecays
import numpy as np
import pandas as pd


def data_fit(data_set, HIGGS_BR, HIGGS_BR_DATA, POLARIZATION):
    def get_br_values(br_dict):
        assert set(data_set.decay_names) == br_dict.keys()
        brs = np.array([br_dict[k] for k in data_set.decay_names])
        return brs

    data_set.data_brs = get_br_values(HIGGS_BR_DATA)
    data_set.fit_start_brs = get_br_values(HIGGS_BR)
    df = pd.DataFrame(index=data_set.decay_names)
    for name, polarization in {
        "default": list(map(float, POLARIZATION.split())),
        "eLpR": (-0.8, 0.3),
        "eRpL": (0.8, -0.3),
        "unpolarized": (0, 0),
    }.items():
        data_set.polarization = polarization
        fit = alldecays.Fit(
            data_set,
            fit_mode="BinomialLeastSquares",
            has_limits=True,
            print_brs_sum_not_1=False,
        )
        df[name] = fit.fit_mode.errors
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
    df = data_fit(data_set, s["HIGGS_BR"], HIGGS_BR_DATA, s["POLARIZATION"])
    df.to_csv(output_file)

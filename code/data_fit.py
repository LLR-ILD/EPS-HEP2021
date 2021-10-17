#!/usr/bin/env python
import alldecays
import numpy as np
import pandas as pd


def data_fit(data_set, HIGGS_BR, HIGGS_BR_DATA):
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
    df = pd.DataFrame(index=fit.fit_mode.parameters)

    df["fit_values"] = fit.fit_mode.values
    df["errors"] = fit.fit_mode.errors
    df["starting_values"] = fit._data_set.fit_start_brs
    df["data_values"] = data_set.data_brs
    covariance = fit.fit_mode.covariance
    for i, cov_partner in enumerate(fit.fit_mode.parameters):
        df[cov_partner] = covariance[i]

    df["toy_values"] = fit.fill_toys(
        n_toys=1, rng=np.random.default_rng(seed=1)
    ).physics[0]
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
    df = data_fit(data_set, s["HIGGS_BR"], HIGGS_BR_DATA)
    df.to_csv(output_file)

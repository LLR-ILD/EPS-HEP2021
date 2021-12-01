#!/usr/bin/env python
import alldecays
import numpy as np
import pandas as pd


def data_fit(data_set, HIGGS_BR, HIGGS_BR_DATA, with_bkg_free=True):
    def get_br_values(br_dict):
        assert set(data_set.decay_names) == br_dict.keys()
        brs = np.array([br_dict[k] for k in data_set.decay_names])
        return brs

    data_set.data_brs = get_br_values(HIGGS_BR_DATA)
    data_set.fit_start_brs = get_br_values(HIGGS_BR)
    df = pd.DataFrame(index=data_set.decay_names)
    original_signal_scaler = data_set.signal_scaler
    for name, scaler in {
        "default": 1,
        r"$ \div 1.5 $": 1 / 1.5,
        "$ \\times 1.5 $": 1.5,
        "$ \\times 2 $": 2,
        "$ \\times 4 $": 4,
        "$ \\times 8 $": 8,
        "$ \\times 16 $": 16,
    }.items():
        data_set.signal_scaler = original_signal_scaler * scaler
        fit = alldecays.Fit(
            data_set,
            fit_mode="BinomialLeastSquares",
            has_limits=True,
            print_brs_sum_not_1=False,
        )
        df[name] = fit.fit_mode.errors

    if with_bkg_free:
        data_set.signal_scaler = original_signal_scaler
        for channel in data_set.get_channels().values():
            channel.bkg_cs_default = np.zeros_like(channel.bkg_cs_default)
        fit = alldecays.Fit(
            data_set,
            fit_mode="BinomialLeastSquares",
            has_limits=True,
            print_brs_sum_not_1=False,
        )
        df.insert(1, "$ \\times 1 $, no bkg", fit.fit_mode.errors)
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

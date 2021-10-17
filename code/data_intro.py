#!/usr/bin/env python
import alldecays
import numpy as np


def data_intro(channel, PROCESS_GROUPS, HIGGS_BR, with_failed_presel=True):
    process_grouping = {v: k for k, values in PROCESS_GROUPS.items() for v in values}
    expected_per_class = alldecays.plotting.util.get_expected_matrix(channel)
    expected_per_class = expected_per_class.transpose()
    if with_failed_presel:
        cs_signal = channel.data_brs * channel.signal_cs_default * channel.signal_scaler
        cs = (
            np.concatenate([cs_signal, channel.bkg_cs_default]) * channel.luminosity_ifb
        )
        expected_per_class["failed_presel"] = cs - expected_per_class.sum(axis=1)
    expected_per_class = expected_per_class.groupby(process_grouping).sum()
    sorted_bkg = sorted(id for id in expected_per_class.index if id not in HIGGS_BR)
    reordered_index = list(HIGGS_BR) + sorted_bkg
    expected_per_class = expected_per_class.reindex(reordered_index)
    expected_per_class["is_higgs_decay"] = [
        id in HIGGS_BR for id in expected_per_class.index
    ]
    return expected_per_class


if __name__ == "__main__":
    import sys

    from helper_data import get_data_set, read_settings

    output_file, input_dir, s = read_settings(sys.argv)
    data_set = get_data_set(input_dir, **s)
    channel_name = output_file.stem.replace("counts_", "")
    channel = data_set.get_channels()[channel_name]
    df = data_intro(channel, s["PROCESS_GROUPS"], s["HIGGS_BR"])
    df.to_csv(output_file)

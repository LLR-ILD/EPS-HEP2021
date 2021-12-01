from pathlib import Path

import alldecays
import numpy as np
import yaml


def read_settings(argv):
    output_file = Path(argv[1])
    if output_file.suffix != ".csv":
        raise Exception(f"Not a valid csv data file: {output_file}.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    input_dir = Path(argv[2])
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {input_dir}.")
    settings = {}
    for arg in argv[3:]:
        file_name = Path(arg)
        if file_name.samefile(__file__):
            continue
        with Path(arg).open() as f:
            settings[file_name.stem] = yaml.safe_load(f)
    return output_file, input_dir, settings


def get_data_set(DATA_DIR, **kw):
    necessary_kw = {"HIGGS_BR", "POLARIZATION", "SIGNAL_SCALER", "LUMINOSITY"}
    if not necessary_kw.issubset(kw):
        raise Exception(f"Missing keys: {necessary_kw - set(kw)}")

    assert DATA_DIR.exists()
    data_set = alldecays.DataSet(
        decay_names=list(kw["HIGGS_BR"].keys()),
        polarization=list(map(float, kw["POLARIZATION"].split())),
        ignore_limited_mc_statistics_bias=True,
    )
    for channel_name in ["e1e1", "e2e2"]:
        data_set.decay_names = [
            f"P{channel_name}h_{dec}" for dec in data_set.decay_names
        ]
        data_set.add_channel(channel_name, DATA_DIR / f"tables_{channel_name}")
        data_set.decay_names = list(kw["HIGGS_BR"].keys())
    data_set.signal_scaler = kw["SIGNAL_SCALER"]
    data_set.luminosity_ifb = kw["LUMINOSITY"]
    data_set.data_brs = np.array(list(kw["HIGGS_BR"].values()))
    data_set.fit_start_brs = data_set.data_brs
    return data_set


if __name__ == "__main__":
    import sys

    print(read_settings(sys.argv))

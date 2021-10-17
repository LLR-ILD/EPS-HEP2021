#!/usr/bin/env python
import typing

import yaml

HIGGS_BR = {
    # "ss":     0.00034,
    "cc": 0.02718,
    "bb": 0.57720,
    "e2e2": 0.00030,
    "e3e3": 0.06198,
    "az": 0.00170,
    "gg": 0.08516 + 0.00034,
    "aa": 0.00242,
    "zz": 0.02616,
    "ww": 0.21756,
}
HIGGS_BR = dict(sorted(HIGGS_BR.items(), key=lambda item: item[1])[::-1])
assert abs(1.0 - sum(HIGGS_BR.values())) < 1e-5

brs_changed = {}
HIGGS_BR_CHANGED_bbww = {k: v for k, v in HIGGS_BR.items()}
HIGGS_BR_CHANGED_bbww["bb"] = HIGGS_BR_CHANGED_bbww["bb"] - 0.15
HIGGS_BR_CHANGED_bbww["ww"] = HIGGS_BR_CHANGED_bbww["ww"] + 0.15
brs_changed["HIGGS_BR_CHANGED_bbww"] = HIGGS_BR_CHANGED_bbww

mumu_target = 0.01
rescaler = (1 - mumu_target) / (1 - HIGGS_BR["e2e2"])
HIGGS_BR_CHANGED_e2e2 = {k: v * rescaler for k, v in HIGGS_BR.items()}
HIGGS_BR_CHANGED_e2e2["e2e2"] = mumu_target
brs_changed["HIGGS_BR_CHANGED_e2e2"] = HIGGS_BR_CHANGED_e2e2
assert abs(1.0 - sum(HIGGS_BR_CHANGED_e2e2.values())) < 1e-5


def grouping_rule(process):
    if process[1:3] in ["2f", "4f"]:
        endings = {"l": "leptonic", "sl": "semileptonic", "h": "hadronic"}
        return f"{process[1:3]} {endings[process.split('_')[-1]]}"
    elif process.startswith("Pqqh_"):
        return "other higgs"
    elif process.startswith("Pe1e1h_") or process.startswith("Pe2e2h_"):
        h_decay = process.split("_")[-1]
        assert h_decay in HIGGS_BR
        return h_decay
    else:
        return process


bkgs = [
    "P2f_z_l",
    "P4f_sw_l",
    "P4f_sw_sl",
    "P4f_sze_l",
    "P4f_szeorsw_l",
    "P4f_sznu_l",
    "P4f_ww_l",
    "P4f_zz_l",
    "P4f_zz_sl",
    "P4f_zzorww_l",
    "Pqqh_zz",
    "Pe2e2h",  # For preselection_plots.py
]
signals = [
    f"{prefix}_{ending}" for prefix in ["Pe1e1h", "Pe2e2h"] for ending in HIGGS_BR
]
PROCESS_GROUPS: typing.Dict[str, typing.List[str]] = {}
for process in signals + list(HIGGS_BR) + bkgs:
    group_name = grouping_rule(process)
    if group_name not in PROCESS_GROUPS:
        PROCESS_GROUPS[group_name] = list()
    PROCESS_GROUPS[group_name].append(process)
# Sort keys and value for convenience.
for k, v in PROCESS_GROUPS.items():
    PROCESS_GROUPS[k] = sorted(v)
PROCESS_GROUPS = {k: PROCESS_GROUPS[k] for k in sorted(PROCESS_GROUPS)}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        raise Exception("Must have exactly one argument: the file to build.")
    setting_to_redo = sys.argv[1]
    setting_val: typing.Dict[str, typing.Any]
    if setting_to_redo == "HIGGS_BR":
        setting_val = HIGGS_BR
    elif setting_to_redo == "PROCESS_GROUPS":
        setting_val = PROCESS_GROUPS
    elif (
        setting_to_redo.startswith("HIGGS_BR_CHANGED")
        and setting_to_redo in brs_changed
    ):
        setting_val = brs_changed[setting_to_redo]
    else:
        raise Exception(f"Invalid command line argument: {setting_to_redo}")

    with (Path(__file__).parent / setting_to_redo).open("w") as f:
        yaml.dump(setting_val, f, sort_keys=False)

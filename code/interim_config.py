"""Use this until we have more time for refactoring."""
from pathlib import Path

import matplotlib.pyplot as plt
from alldecays.plotting.util import get_experiment_tag

LUMINOSITY = 2000
POLARIZATION = (-0.8, 0.3)
SIGNAL = "Pe2e2h"
SIGNAL_SCALER = 1
IMG_PATH = Path(__file__).parent.parent / "img/code"
IMG_PATH.mkdir(exist_ok=True)


signal_endings = {
    "aa": "H→γγ",
    "az": "H→Zγ",
    "bb": "H→bb",
    "cc": "H→cc",
    "e2e2": "H→μμ",
    "e3e3": "H→ττ",
    "gg": "H→gg",
    "ww": "H→WW",
    "zz": "H→ZZ",
}


def grouping_rule(process):
    if process[1:3] in ["2f", "4f"]:
        endings = {"l": "leptonic", "sl": "semileptonic", "h": "hadronic"}
        return f"{process[1:3]} {endings[process.split('_')[-1]]}"
    elif process.startswith("Pqqh_"):
        return "other higgs"
    elif process.startswith("Pe1e1h_") or process.startswith("Pe2e2h_"):
        return signal_endings[process.split("_")[-1]]
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
    f"{prefix}_{ending}" for prefix in ["Pe1e1h", "Pe2e2h"] for ending in signal_endings
]
PROCESS_GROUPS = {
    process: grouping_rule(process)
    for process in signals + list(signal_endings.values()) + bkgs
}

_BRS = {
    # "H→ss":     0.00034,
    "H→cc": 0.02718,
    "H→bb": 0.57720,
    "H→μμ": 0.00030,
    "H→ττ": 0.06198,
    "H→Zγ": 0.00170,
    "H→gg": 0.08516 + 0.00034,
    "H→γγ": 0.00242,
    "H→ZZ": 0.02616,
    "H→WW": 0.21756,
}
_BRS = dict(sorted(_BRS.items(), key=lambda item: item[1])[::-1])


def channel_decay_names(channel):
    dec_rename = {
        "H→cc": "cc",
        "H→bb": "bb",
        "H→μμ": "e2e2",
        "H→ττ": "e3e3",
        "H→Zγ": "az",
        "H→gg": "gg",
        "H→γγ": "aa",
        "H→ZZ": "zz",
        "H→WW": "ww",
    }
    return [f"P{channel}h_" + dec_rename[k] for k in _BRS]


def add_ild(ax):
    get_experiment_tag("ILD_preliminary")(ax)


def save_hook(fig, name, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    fig.savefig(IMG_PATH / f"{name}.pdf", dpi=300, transparent=True)
    plt.close(fig)

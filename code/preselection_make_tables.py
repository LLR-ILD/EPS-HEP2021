"""Create higgstables-like tables for the preselection steps.

Some patching around the higgstables capabilities. This creates the input data
for the preselection plots of the presentation.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import higgstables
import numexpr
import numpy as np
import pandas as pd
import uproot


class FileToDF(higgstables.FileToCounts):
    def __init__(self, rootfile_path, config, polarization, signal_process):
        self._rootfile_path = rootfile_path
        self._config = config
        self._polarization = polarization

        self._rootfile = uproot.open(self._rootfile_path)
        self.name = self._rootfile_path.absolute().parent.name
        self.n_not_triggered = self.run_triggers()

        # Needed in superclass.
        self._loaded_arrays = defaultdict(dict)

    def get_preselection_variables(self):
        assert len(self._config.preselections) == 1
        presel = self._config.preselections[0]
        return presel.variables

    def get_df(self):
        assert len(self._config.preselections._triggers) == 1
        for presel in self._config.preselections:
            break  # There is only one element.
        df = pd.DataFrame(self._get_array_dict(presel))
        df.insert(0, "polarization", self._polarization)
        df.insert(0, "name", self.name)
        return df


def get_files(root_dir, config):
    def _apply_ignoring(path_set):
        ignored_paths = [
            path
            for path in path_set
            if path.absolute().parent.name in config.ignored_processes
        ]
        for ignored_path in ignored_paths:
            path_set.remove(ignored_path)
        return path_set

    for table_name, search_pattern in config.tables.items():
        in_this_table = set(Path(root_dir).glob(search_pattern))
        in_this_table = _apply_ignoring(in_this_table)
        yield table_name, in_this_table


def get_triggered_events(root_dir, config_file, signal_process):
    config = higgstables.config.load_config.load_config(config_file)
    dfs = []
    n_not_triggered = defaultdict(dict)
    for polarization, files in get_files(root_dir, config):
        for rootfile in files:
            file_to_df = FileToDF(rootfile, config, polarization, signal_process)
            dfs.append(file_to_df.get_df())
            name = file_to_df.name
            n_not_triggered[polarization][name] = file_to_df.n_not_triggered
    triggered_events = pd.concat(dfs)
    n_not_triggered = pd.DataFrame(n_not_triggered)
    return triggered_events, n_not_triggered


def get_hist_per_step(root_dir, config_file, table_dir, bins=100):
    table_dir = table_dir / f"presel_{root_dir.stem.split('_')[-1]}"
    signal_process = f"P{root_dir.stem.split('_')[-1]}h"
    events, n_not_triggered = get_triggered_events(
        root_dir, config_file, signal_process
    )
    conf_dict = higgstables.config.load_config._load_config_dict(config_file)
    steps = conf_dict["higgstables"]["preselections"][0]["condition"]
    step_info = []
    for i, step in enumerate(steps, start=1):
        step_vars = set(higgstables.config.util.get_variables_from_expression(step))
        if len(step_vars) == 1:
            name = step_vars.pop()
            events["x"] = events[name]
        else:
            pattern = re.compile(">|<|>=|<=|==")
            name = pattern.split(step)[0]
            events["x"] = numexpr.evaluate(name, events)
        binning = np.linspace(min(events["x"]), max(events["x"]), bins + 1)
        for polarization, pol_df in events.groupby("polarization"):
            pure_pol_dir = Path(table_dir) / polarization
            pure_pol_dir.mkdir(exist_ok=True, parents=True)
            bin_counts = {}
            for process, df in pol_df.groupby("name"):
                bin_counts[process] = np.histogram(df["x"], binning)[0]
            step_df = pd.DataFrame(bin_counts)
            step_df = step_df.reindex(sorted(step_df.columns), axis=1)
            step_df.to_csv(pure_pol_dir / f"step_{i}.csv", index=False)
        events = events[numexpr.evaluate(step, events)]
        step_info.append({})
        step_info[-1]["index"] = f"step_{i}"
        step_info[-1]["name"] = step
        step_info[-1]["min"] = binning[0]
        step_info[-1]["max"] = binning[-1]
        step_info[-1]["n_bins"] = bins
    step_info = pd.DataFrame(step_info)
    step_info = step_info.set_index("index")
    step_info = step_info.sort_index()
    step_info.to_csv(table_dir / "step_info.csv")
    n_not_triggered = n_not_triggered.sort_index()
    n_not_triggered.to_csv(table_dir / "n_not_triggered.csv")

    n_after_presel = events.groupby("name")["polarization"].value_counts()
    n_after_presel = n_after_presel.unstack(level="polarization")
    n_after_presel.fillna(0)
    n_after_presel = n_after_presel.sort_index()
    n_after_presel.to_csv(table_dir / "n_after_preselection.csv")


if __name__ == "__main__":
    root_dir = Path(sys.argv[1])
    config_file = Path(sys.argv[2])
    table_dir = Path(sys.argv[3])
    get_hist_per_step(root_dir, config_file, table_dir)

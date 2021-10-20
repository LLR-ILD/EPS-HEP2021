from collections import defaultdict
from pathlib import Path

import higgstables
import numpy as np
import pandas as pd


def get_presel_process_grouping(PROCESS_GROUPS):
    # In the preselection step, we are not interested in the decay mode.
    presel_process_groups = {
        v: k for k, values in PROCESS_GROUPS.items() for v in values
    }
    for process in presel_process_groups:
        if process.startswith("Pe1e1h") or process.startswith("Pe2e2h"):
            presel_process_groups[process] = process[: len("Pe1e1h")]
    return presel_process_groups


class WeightedPreselectionTables:
    def __init__(self, presel_table_dir, config_file, polarization, luminosity):
        self._presel_table_dir = presel_table_dir
        self._polarization = polarization
        self._luminosity = luminosity
        self.name = presel_table_dir.stem[len("presel_") :]
        self._n_simulated = self.get_n_simulated()
        self.weight = self.get_weight(config_file, self._n_simulated)

        self.step_info = pd.read_csv(presel_table_dir / "step_info.csv", index_col=0)
        self._fill_step_tables()

    def get_n_not_triggered(self):
        path = self._presel_table_dir / "n_not_triggered.csv"
        n_raw = pd.read_csv(path, index_col=0)
        n_raw = n_raw.fillna(0)
        n_weighted = n_raw * self.weight
        return n_weighted.sum(axis=1)

    def get_n_after_preselection(self):
        path = self._presel_table_dir / "n_after_preselection.csv"
        n_raw = pd.read_csv(path, index_col=0)
        n_raw = n_raw.fillna(0)
        n_weighted = n_raw * self.weight
        return n_weighted.sum(axis=1)

    def get_n_simulated(self):
        path = self._presel_table_dir / "n_not_triggered.csv"
        n_not_triggered = pd.read_csv(path, index_col=0)
        n_not_triggered = n_not_triggered.fillna(0)
        n_triggered = self._collect_counts(
            n_not_triggered.index, n_not_triggered.columns
        )
        n_simulated = n_triggered + n_not_triggered
        return n_simulated

    def get_weight(self, config_file, n_simulated):
        n_expected_dict = self.get_n_expected_dict(config_file)
        processes = n_simulated.index
        df_dict = {}
        for pol in n_simulated.columns:
            proc_dict = n_expected_dict[pol]
            df_dict[pol] = {k: v for k, v in proc_dict.items() if k in processes}
        n_expected = pd.DataFrame(df_dict)
        n_expected = n_expected.fillna(0)
        n_simulated = n_simulated.replace(0, -1)  # Avoid 0/0 division.
        return n_expected / n_simulated

    def _collect_counts(self, index, columns, step_name="step_1.csv"):
        counts = pd.DataFrame(index=index)
        for pol in columns:
            bin_counts = pd.read_csv(self._presel_table_dir / pol / step_name)
            assert set(bin_counts.columns).issubset(counts.index)
            counts[pol] = bin_counts.sum(axis="rows")
        counts = counts.fillna(0)
        return counts

    def get_n_expected_dict(self, config_file):
        config = higgstables.config.load_config.load_config(config_file)
        cs_dict = config.cross_sections.polarization_weighted(self._polarization)
        n_expected = defaultdict(dict)
        for pol in cs_dict:
            for process in cs_dict[pol]:
                n_expected[pol][process] = self._luminosity * cs_dict[pol][process]
        return n_expected

    def _fill_step_tables(self):
        self._steps = {}
        for step_id in self.step_info.index:
            info = self.step_info.loc[step_id]
            weighted_counts = pd.DataFrame(
                0,
                columns=self.weight.index,
                index=np.arange(info["n_bins"]),
            )
            for pol in self.weight.columns:
                df = pd.read_csv(self._presel_table_dir / pol / f"{step_id}.csv")
                df = df.mul(self.weight[pol]).fillna(0)
                weighted_counts = weighted_counts + df
            self._steps[step_id] = weighted_counts

    def iter_steps(self):
        for step_id, step_table in self._steps.items():
            info = self.step_info.loc[step_id]
            binning = np.linspace(info["min"], info["max"], info["n_bins"] + 1)
            yield step_table, info["name"], binning

    @property
    def n_steps(self):
        return len(self._steps)


def get_eff_pur_table(presel_tables: WeightedPreselectionTables):
    signal_process = f"P{presel_tables.name}h"
    n_after = presel_tables.get_n_after_preselection()
    all_processes = n_after.index
    signal_columns = [c for c in all_processes if c.startswith(signal_process)]
    # Get signal and total counts from tables.
    n_signal_list = []
    n_total_list = []
    for step_table, _, _ in presel_tables.iter_steps():
        n_signal_list.append(step_table[signal_columns].sum().sum())
        n_total_list.append(step_table.sum().sum())
    # The data point after the last selection step is still missing.
    n_signal_list.append(n_after[signal_columns].sum().sum())
    n_total_list.append(n_after.sum().sum())

    n_signal = np.array(n_signal_list)
    n_total = np.array(n_total_list)
    n_signal_untriggered = presel_tables.get_n_not_triggered()[signal_columns].sum()

    efficiency = n_signal / (n_signal[0] + n_signal_untriggered)
    purity = n_signal / n_total
    df = pd.DataFrame(dict(efficiency=efficiency, purity=purity))
    return df


def get_step_dfs(presel_tables: WeightedPreselectionTables, PROCESS_GROUPS: dict):
    process_grouping = get_presel_process_grouping(PROCESS_GROUPS)
    step_dfs = []
    for step_table, _, _ in presel_tables.iter_steps():
        step_table = step_table.transpose()
        step_table = step_table.groupby(process_grouping).sum()
        step_table = step_table.transpose()
        step_dfs.append(step_table)
    return step_dfs


if __name__ == "__main__":
    import sys

    from helper_data import read_settings

    argv = sys.argv
    config_file = argv.pop(3)
    # argv[2] should be the data folder, not a file.
    argv[2] = str(Path(sys.argv[2]).parents[1])
    output_file, input_dir, s = read_settings(argv)  #

    polarization = list(map(float, s["POLARIZATION"].split()))
    wpt = WeightedPreselectionTables(
        input_dir,
        config_file,
        polarization=polarization,
        luminosity=float(s["LUMINOSITY"]),
    )
    if "step" in output_file.stem:
        name = output_file.stem
        input_step_number = name[name.find("step") + len("step") :].split("_")[0]
        step_phrase = f"step{input_step_number}"
        assert input_step_number.isdigit()
        step_dfs = get_step_dfs(wpt, s["PROCESS_GROUPS"])
        for i, step_df in enumerate(step_dfs, 1):
            step_df.to_csv(
                str(output_file).replace(step_phrase, f"step{i}"), index=False
            )
    elif "eff_pur" in output_file.stem:
        df = get_eff_pur_table(wpt)
        df.to_csv(output_file, index=False)
    else:
        raise Exception(output_file)

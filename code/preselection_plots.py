import re
import sys
from collections import defaultdict
from pathlib import Path

import higgstables
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interim_config import LUMINOSITY, POLARIZATION, PROCESS_GROUPS, add_ild, save_hook

# In the preselection step, we are not interested in the decay mode.
presel_process_groups = {k: v for k, v in PROCESS_GROUPS.items()}
for process in presel_process_groups:
    if process.startswith("Pe1e1h") or process.startswith("Pe2e2h"):
        presel_process_groups[process] = process[: len("Pe1e1h")]


class WeightedPreselectionTables:
    def __init__(self, presel_table_dir, config_file):
        self._presel_table_dir = presel_table_dir
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
        cs_dict = config.cross_sections.polarization_weighted(POLARIZATION)
        n_expected = defaultdict(dict)
        for pol in cs_dict:
            for process in cs_dict[pol]:
                n_expected[pol][process] = LUMINOSITY * cs_dict[pol][process]
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


def plot_efficiency(presel_tables: WeightedPreselectionTables):
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

    x = np.arange(len(n_signal))
    efficiency = n_signal / (n_signal[0] + n_signal_untriggered)
    purity = n_signal / n_total
    eff_pur = efficiency * purity
    for i in x:
        fig, ax = plt.subplots(figsize=(4, 3))
        add_ild(ax)
        ax.plot(x[: i + 1], efficiency[: i + 1], label="efficiency", marker="o")
        ax.plot(x[: i + 1], purity[: i + 1], label="purity", marker="d")
        ax.plot(x[: i + 1], eff_pur[: i + 1], label="eff * pur", marker="*")
        ax.set_xticks(x)
        ax.set_xlabel("step")
        ax.set_xlim((min(x) - 0.5, max(x) + 0.5))
        ax.set_ylim((0, 1))
        ax.legend(bbox_to_anchor=(1, 0.9), loc="upper right")
        save_hook(fig, f"presel_{presel_tables.name}_eff_{i}")
    return dict(efficiency=efficiency, purity=purity)


def get_latex(string):
    """Quick fix for nicer figure texts."""
    for normal, latex in {
        "m_z": "$M_Z$ [GeV]",
        "m_recoil": r"$M_{\mathrm{recoil}}$ [GeV]",
        "abs(cos_theta_z)": "$| \\mathrm{cos} \\theta_Z |$",
        "abs(cos_theta_miss)": "$| \\mathrm{cos} \\theta_{\\mathrm{miss}} |$",
        "abs(cos_theta_z - cos_theta_miss)": "$| \\mathrm{cos} \\theta_Z - \\mathrm{cos} \\theta_{\\mathrm{miss}} |$",
        "cos_theta_z": "$\\mathrm{cos} \\theta_Z$",
        "cos_theta_miss": "$\\mathrm{cos} \\theta_{\\mathrm{miss}}$",
    }.items():
        if normal in string:
            string = string.replace(normal, latex)
    return string


def plot_var(step_table, step_name, binning, signal_process, ax, group_order=None):
    if group_order is None:
        columns = {v for k, v in presel_process_groups.items() if k in step_table}
    else:
        columns = group_order
    grouped_table = pd.DataFrame(0, index=step_table.index, columns=columns)
    for process_name in step_table.columns:
        group_name = presel_process_groups[process_name]
        grouped_table[group_name] += step_table[process_name]

    if group_order is None:
        group_order = list(grouped_table.sum().sort_values(ascending=False).index)
        # Move signal process to front.
        group_order.insert(0, group_order.pop(group_order.index(signal_process)))

    x = (binning[1:] + binning[:-1]) / 2
    kw = dict(
        width=binning[1:] - binning[:-1],
        alpha=1,
    )
    bottom = np.zeros_like(grouped_table[signal_process])
    for group in group_order:
        ax.bar(x, grouped_table[group], bottom=bottom, label=group, **kw)
        bottom += grouped_table[group]
        kw["alpha"] = 0.5  # First group was signal, now bkg groups.

    # Add horizontal lines
    for cut_value in re.findall(r"[-+]?\d*\.\d+|\d+", step_name):
        ax.axvline(float(cut_value), ls="--", color="black")
        if step_name.startswith("abs(") and min(binning) < 0:
            ax.axvline(-float(cut_value), ls="--", color="black")
    return group_order


def plot_preselection_steps(presel_tables: WeightedPreselectionTables):
    signal_process = f"P{presel_tables.name}h"
    nrows = 1 + (presel_tables.n_steps - 1) // 2
    var_fig, axs = plt.subplots(ncols=2, nrows=nrows, figsize=(10, 4 * nrows))
    axs = axs.flatten()
    proc_args = dict(ncols=3, nrows=2, figsize=(15, 4 * 2))
    fig_proceedings, axs_proceedings = plt.subplots(**proc_args)
    axs_proceedings = axs_proceedings.flatten()
    group_order = None
    for i, (step_table, step_name, binning) in enumerate(
        presel_tables.iter_steps(), start=1
    ):
        step_vars = set(
            higgstables.config.util.get_variables_from_expression(step_name)
        )
        if len(step_vars) == 1:
            var_name = step_vars.pop()
        else:
            pattern = re.compile(">|<|>=|<=|==")
            var_name = pattern.split(step_name)[0]
        for ax in [axs[i - 1], axs_proceedings[i - 1]]:
            group_order = plot_var(
                step_table,
                step_name,
                binning,
                signal_process,
                ax,
                group_order,
            )
            ax.set_xlabel(get_latex(var_name), fontsize=18)
            ax.set_label("weighted bin counts")
            ax.set_title(f"step {i}: {get_latex(step_name)}")
    for ax in [axs[0], axs_proceedings[0]]:
        ax.legend(bbox_to_anchor=(1, 0.95), loc="upper right")
        add_ild(ax)
    var_fig.tight_layout()
    for ax in axs:
        ax.set_visible(False)
    for i in range(presel_tables.n_steps):
        axs[i].set_visible(True)
        save_hook(var_fig, f"presel_{presel_tables.name}_{i + 1}", tight_layout=False)
    return fig_proceedings


def plot_preselection_for_proceedings(
    presel_tables: WeightedPreselectionTables,
    fig_proceedings,
    eff_pur_dict,
):
    efficiency = eff_pur_dict["efficiency"]
    purity = eff_pur_dict["purity"]
    eff_pur = efficiency * purity
    x = np.arange(len(efficiency))
    ax = fig_proceedings.get_axes()[-1]
    ax.plot(x, efficiency, label="efficiency", marker="o")
    ax.plot(x, purity, label="purity", marker="d")
    ax.plot(x, eff_pur, label="eff * pur", marker="*")
    ax.set_xticks(x)
    ax.set_xlabel("step", fontsize=18)
    ax.set_xlim((min(x) - 0.5, max(x) + 0.5))
    ax.set_ylim((0, 1))
    ax.legend(bbox_to_anchor=(1, 0.9), loc="upper right")
    save_hook(fig_proceedings, f"presel_{presel_tables.name}_for_proceedings")


if __name__ == "__main__":
    table_dir = sys.argv[1]
    config_file = sys.argv[2]
    found_a_presel = False
    for presel_table_dir in Path(table_dir).iterdir():
        if presel_table_dir.stem.startswith("presel_"):
            found_a_presel = True
            wpt = WeightedPreselectionTables(presel_table_dir, config_file)
            eff_pur_dict = plot_efficiency(wpt)
            fig_proceedings = plot_preselection_steps(wpt)
            plot_preselection_for_proceedings(wpt, fig_proceedings, eff_pur_dict)
    assert found_a_presel

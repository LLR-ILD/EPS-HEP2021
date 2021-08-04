"""This file  shows a typical setup procedure.
"""
import sys
from pathlib import Path

import alldecays
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interim_config import (
    _BRS,
    IMG_PATH,
    LUMINOSITY,
    POLARIZATION,
    PROCESS_GROUPS,
    SIGNAL,
    SIGNAL_SCALER,
    add_ild,
    channel_decay_names,
    save_hook,
)

get_fit_first_call = True


def get_fit(ds):
    get_fit_first_call = False
    return alldecays.Fit(
        ds,
        fit_mode="BinomialLeastSquares",
        print_brs_sum_not_1=get_fit_first_call,
        has_limits=True,
    )


def get_leptonic_ds(data_dir):
    assert data_dir.exists()
    data_set = alldecays.DataSet(
        decay_names=list(_BRS.keys()),
        polarization=POLARIZATION,
        ignore_limited_mc_statistics_bias=True,
    )
    for channel_name in ["e1e1", "e2e2"]:
        data_set.decay_names = channel_decay_names(channel_name)
        data_set.add_channel(channel_name, data_dir / f"tables_{channel_name}")
    data_set.decay_names = list(_BRS.keys())
    data_set.signal_scaler = SIGNAL_SCALER
    data_set.luminosity_ifb = LUMINOSITY
    data_set.data_brs = np.array(list(_BRS.values()))
    data_set.fit_start_brs = data_set.data_brs
    return data_set


def plot_counts_in_samples(ds):
    fig, ax = plt.subplots(figsize=(4, 4))
    add_ild(ax)
    df = pd.DataFrame()
    for channel_name, channel in ds.get_channels().items():
        if channel_name != SIGNAL[1:-1]:
            continue
        expected = alldecays.plotting.util.get_expected_matrix(channel)
        expected_per_process = expected.sum(axis=0)
        assert set(expected_per_process.index).issubset(PROCESS_GROUPS)
        expected_per_group = expected_per_process.groupby(PROCESS_GROUPS).sum()
        if channel_name == "e1e1":
            channel_name = "$Z \\to ee$"
        if channel_name == "e2e2":
            channel_name = "$Z \\to \\mu\\mu$"
        df[channel_name] = expected_per_group
    df = df.reindex(list(_BRS) + sorted(id for id in df.index if id not in _BRS))
    df = df.transpose()
    left = np.zeros(len(df))
    for col in df.columns:
        ax.barh(
            np.arange(len(df)),
            df[col],
            left=left,
            label=col,
            alpha=1 if col in _BRS else 0.5,
        )
        left += df[col]
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("expected signal counts")
    ax.legend(loc="center right")
    save_hook(fig, "intro_sample_counts")


def plot_counts_in_categories(ds):
    channel = ds.get_channels()[SIGNAL[1:-1]]
    expected = alldecays.plotting.util.get_expected_matrix(channel)
    data_counts = expected.sum(axis=1)
    for add_data_dots in [True, False]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim((0, 1.1 * max(data_counts)))
        add_ild(ax)
        alldecays.plotting.box_counts(channel, ax, no_bkg=True)
        if add_data_dots:
            ax.scatter(
                data_counts,
                np.arange(len(data_counts)),
                label="including bkg",
                color="black",
            )
            ax.legend(loc="center right")
            save_hook(fig, "intro_category_counts_w_bkg")
        else:
            save_hook(fig, "intro_category_counts")

    fig, ax = plt.subplots(figsize=(4, 4))
    add_ild(ax)
    alldecays.plotting.box_counts(channel, ax, no_bkg=True, is_normalized=True)
    ax.get_legend().remove()
    save_hook(fig, "intro_signal_composition_per_category")


def plot_channel_for_intro(table_dir):
    ds = get_leptonic_ds(table_dir)
    plot_counts_in_samples(ds)
    plot_counts_in_categories(ds)


def plot_matrices(table_dir):
    ds = get_leptonic_ds(table_dir)
    channel = ds.get_channels()[SIGNAL[1:-1]]

    fig, ax = plt.subplots(figsize=(8, 10))
    add_ild(ax)
    alldecays.plotting.probability_matrix(channel, ax=ax, no_bkg=True)
    save_hook(fig, "probability_matrix")

    for no_bkg in [
        # True
        False,
    ]:
        x_len = len(channel.decay_names) + 3
        name = "expected_counts_matrix"
        if not no_bkg:
            x_len += len(channel.bkg_names)
            name += f"_bkg_{SIGNAL[1:-1]}"
        fig, ax = plt.subplots(figsize=(x_len / 3 * 2, 10))
        add_ild(ax)
        alldecays.plotting.expected_counts_matrix(channel, ax=ax, no_bkg=no_bkg)
        save_hook(fig, name)


def make_bias_table(fit, table_name="bias_table"):
    def h_to_latex(h_str):
        return (
            "$"
            + (h_str)
            .replace("→μμ", "\to \\mu\\mu")
            .replace("→ττ", "\to \tau\tau")
            .replace("→Zγ", "\to Z\\gamma")
            .replace("→ZZ*", "\to ZZ^*")
            .replace("→γγ", "\to \\gamma\\gamma")
            .replace("→", "\to ")
            + "$"
        )

    fp = alldecays.plotting.util.get_fit_parameters(fit, param_space="physics")
    pd.DataFrame(
        {
            "SM BR": 100 * fp.starting_values,
            # "minimum": 100 * fp.values,  # TODO: Put back in for MC bias.
            r"$\sigma_{\mathrm{stat}}$": 100 * fp.errors,
        },
        index=list(map(h_to_latex, fp.names)),
    ).to_latex(buf=(IMG_PATH / f"{table_name}.tex"), float_format="%0.2f", escape=False)


def plot_fits(table_dir):
    ds = get_leptonic_ds(table_dir)
    fit = get_fit(ds)
    fig = alldecays.plotting.compare_values({"Fit result": fit})
    ax = fig.get_axes()[0]
    add_ild(ax)
    save_hook(fig, "br_estimates")

    fig, ax = plt.subplots(figsize=(4, 3))
    alldecays.plotting.compare_errors_only({"Fit result": fit}, ax=ax)
    add_ild(ax)
    ax.set_ylabel(r"$|\Delta_X| = |g_X / g_X^{SM} - 1|$")
    # Manually create an inset.
    ax_inset = ax.inset_axes([0.15, 0.3, 0.6, 0.5])
    n_small = 5
    x, y = ax.collections[0].get_offsets()[:n_small].transpose()
    ax_inset.scatter(x, y, color="C1", marker="*")
    ax_inset.plot(x, y, color="C1", alpha=0.3)
    ax_inset.set_xticks([])
    ax_inset.set_ylim((0, 0.2))
    ax.indicate_inset_zoom(ax_inset, edgecolor="grey")
    ax.get_legend().remove()
    save_hook(fig, "br_relative_error")

    fig, ax = plt.subplots(figsize=(5, 5))
    alldecays.plotting.fit_correlations(fit, ax)
    add_ild(ax)
    save_hook(fig, "correlations")

    make_bias_table(fit)


def _plot_toys_helper(fit, n_toys, prefix, keep_only=None):
    def redo_label(label):
        number = label.split(" ")[0]
        if not number.isnumeric():
            return label
        new_number = f"{100 * float(number):.2f}%"
        return label.replace(number, new_number)

    def format_x_percent(x, y):
        if x < 0.1:
            return f"{100 * x:.2f}"
        else:
            return f"{100 * x:.1f}"

    fit.fill_toys(n_toys)
    figs = alldecays.plotting.toy_hists(fit)
    for name, fig in figs.items():
        if keep_only is not None and name not in keep_only:
            continue
        name = name.replace("→", "_")
        ax = fig.get_axes()[0]
        if name == "H_Zγ" and prefix == "toy_":
            # Remove some digits in a crude way.
            handles, labels = ax.get_legend_handles_labels()
            kw = {"handles": handles}
            kw["labels"] = [redo_label(label_text) for label_text in labels]
            kw["title"] = ax.get_legend().get_title().get_text()
            kw["bbox_to_anchor"] = (0.95, 0.9)
            kw["loc"] = "upper right"
            ax.legend(**kw)
        else:
            ax.get_legend().remove()
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_x_percent))
        ax.set_xlabel("BR [%]")
        add_ild(ax)
        save_hook(fig, f"{prefix}{name}")


def plot_toys(table_dir, n_toys):
    ds = get_leptonic_ds(table_dir)
    fit = get_fit(ds)
    _plot_toys_helper(fit, n_toys, "toy_")


def plot_changed_brs(table_dir, n_toys):
    ds = get_leptonic_ds(table_dir)
    data_brs_dict = {k: v for k, v in _BRS.items()}
    data_brs_dict["H→bb"] = data_brs_dict["H→bb"] - 0.15
    data_brs_dict["H→WW"] = data_brs_dict["H→WW"] + 0.15
    data_brs = np.array(list(data_brs_dict.values()))
    ds.data_brs = data_brs
    fit = get_fit(ds)

    fig = alldecays.plotting.compare_values({"Fit result": fit})
    ax = fig.get_axes()[0]
    ax.bar(
        np.arange(len(data_brs)),
        data_brs,
        hatch="//",
        fill=False,
        edgecolor="black",
        label="Data truth",
    )
    add_ild(ax)
    ax.legend(loc="center right")
    save_hook(fig, "changed_br_estimates")

    _plot_toys_helper(fit, n_toys, "changed_toy_", keep_only=["H→bb", "H→WW"])


def plot_projections(table_dir):
    compare_polarizations(table_dir)
    compare_br_scenarios(table_dir)
    compare_signal_scaler(table_dir)
    # Boring: compare_luminosity(table_dir)  # Boring.


def compare_polarizations(table_dir):
    ds = get_leptonic_ds(table_dir)
    fits = {}
    for name, pol in {
        "eLpR": (-0.8, 0.3),
        "eRpL": (0.8, -0.3),
        "unpolarized": (0, 0),
    }.items():
        ds.polarization = pol
        fits[name] = get_fit(ds)
    fig, ax = plt.subplots(figsize=(6, 3))
    relative_change_to_standard(table_dir, fits, ax)
    ax.legend(title="Beam polarization", loc="upper left", bbox_to_anchor=(1, 1.05))
    save_hook(fig, "comparison_polarizations")


def compare_signal_scaler(table_dir):
    ds = get_leptonic_ds(table_dir)
    fits_wrong_order, fits = {}, {}
    for name, scaler in {
        r"$ \div 1.5 $": 1 / 1.5,
        "$ \\times 1.5 $": 1.5,
        "$ \\times 2 $": 2,
        "$ \\times 4 $": 4,
    }.items():
        ds.signal_scaler = SIGNAL_SCALER * scaler
        fits_wrong_order[name] = get_fit(ds)

    # Background-free.

    ds.signal_scaler = SIGNAL_SCALER
    for channel in ds.get_channels().values():
        channel.bkg_cs_default = np.zeros_like(channel.bkg_cs_default)
    fits["$ \\times 1 $, no bkg"] = get_fit(ds)
    fits.update(fits_wrong_order)

    fig, ax = plt.subplots(figsize=(6, 3))
    relative_change_to_standard(table_dir, fits, ax)
    ax.legend(
        title=r"$\sigma_{\mathrm{ZH}}$ rescaled",
        loc="upper left",
        bbox_to_anchor=(1, 1.05),
    )
    ylim = ax.get_ylim()
    save_hook(fig, "comparison_signal_scaler")

    fit_parts = {}
    for part in ["$ \\times 1 $, no bkg"]:
        fit_parts[part] = fits[part]
    fig, ax = plt.subplots(figsize=(6, 3))
    relative_change_to_standard(table_dir, fit_parts, ax)
    ax.legend(
        title=r"$\sigma_{\mathrm{ZH}}$ rescaled",
        loc="upper left",
        bbox_to_anchor=(1, 1.05),
    )
    ax.set_ylim(ylim)
    save_hook(fig, "comparison_signal_scaler_partial")


def compare_br_scenarios(table_dir):
    ds = get_leptonic_ds(table_dir)

    fits = {}
    data_brs_dict = {k: v for k, v in _BRS.items()}
    label = "$\\searrow 15 \\%: H \\to bb$\n$\\nearrow 15\\%: H \\to WW$"
    data_brs_dict["H→bb"] = data_brs_dict["H→bb"] - 0.15
    data_brs_dict["H→WW"] = data_brs_dict["H→WW"] + 0.15
    data_brs = np.array(list(data_brs_dict.values()))
    ds.data_brs = data_brs
    fits[label] = get_fit(ds)

    fig, ax = plt.subplots(figsize=(6, 3))
    relative_change_to_standard(table_dir, fits, ax)
    ax.legend(title="Higgs BRs", loc="upper left", bbox_to_anchor=(1, 1.05))
    save_hook(fig, "comparison_br_scenarios_1")

    data_brs_dict = {k: v for k, v in _BRS.items()}
    mumu_target = 0.01
    label = f"$BR(H \\to \\mu \\mu) = {int(100 * mumu_target)} \\%$"
    rescaler = (1 - data_brs_dict["H→μμ"]) / (1 - mumu_target)
    data_brs_dict["H→μμ"] = mumu_target * rescaler
    data_brs = np.array(list(data_brs_dict.values()))
    assert abs(sum(data_brs) - rescaler) < 1e-4
    data_brs = data_brs / sum(data_brs)
    ds.data_brs = data_brs
    ds.signal_scaler = ds.signal_scaler * rescaler
    mumu_fit = get_fit(ds)
    mumu_fp = alldecays.plotting.util.get_fit_parameters(
        mumu_fit, param_space="physics"
    )
    mumu_unc = 100 * mumu_fp.errors[mumu_fp.names.index("H→μμ")]
    label += (
        "\n(then $\\sigma_{\\mathrm{stat}} ^ {\\mu \\mu} = " + fr"{mumu_unc:.1f} \%$)"
    )

    fits[label] = get_fit(ds)

    fig, ax = plt.subplots(figsize=(6, 3))
    relative_change_to_standard(table_dir, fits, ax)
    ax.legend(title="Higgs BRs", loc="upper left", bbox_to_anchor=(1, 1.05))
    save_hook(fig, "comparison_br_scenarios_2")


def compare_luminosity(table_dir):
    ds = get_leptonic_ds(table_dir)
    fits = {}
    fig, ax = plt.subplots(figsize=(6, 3))
    relative_change_to_standard(
        table_dir, fits, ax, change_color=False, change_alpha=True
    )

    lumis = [2, 4, 100]
    for i in lumis:
        ds.luminosity_ifb = LUMINOSITY * i
        fits[f"$\\times {i}$"] = get_fit(ds)
    add_relative_change_to_standard(
        table_dir, fits, ax, change_color=False, change_alpha=True
    )
    ax.legend(title="Luminosity", loc="upper left", bbox_to_anchor=(1, 1.05))

    fits = {}
    for i in lumis:
        ds.luminosity_ifb = LUMINOSITY / i
        fits[f"$\\times {i}$"] = get_fit(ds)
    add_relative_change_to_standard(
        table_dir, fits, ax, change_color=False, change_alpha=True
    )
    save_hook(fig, "comparison_luminosity")


def relative_change_to_standard(
    table_dir, fits, ax, change_color=True, change_alpha=False
):
    add_ild(ax)
    ds = get_leptonic_ds(table_dir)
    standard_fit = get_fit(ds)
    standard_fp = alldecays.plotting.util.get_fit_parameters(
        standard_fit, param_space="physics"
    )
    x = np.arange(len(standard_fp.errors))
    ax.scatter(
        x, np.ones_like(x), marker="*", color="C1", label="Fit described\nin slides"
    )
    add_relative_change_to_standard(
        table_dir, fits, ax, change_color=change_color, change_alpha=change_alpha
    )


def add_relative_change_to_standard(
    table_dir, fits, ax, change_color=True, change_alpha=False
):
    ds = get_leptonic_ds(table_dir)
    standard_fit = get_fit(ds)
    standard_fp = alldecays.plotting.util.get_fit_parameters(
        standard_fit, param_space="physics"
    )
    x = np.arange(len(standard_fp.errors))
    color = "C2"
    alpha = 1
    for i, (fit_name, fit) in enumerate(fits.items()):
        if change_color:
            color = f"C{i+2}"
        if change_alpha:
            alpha = 0.1 + 0.9 * (len(fits) - i - 1) / len(fits)
        fp = alldecays.plotting.util.get_fit_parameters(fit, param_space="physics")
        y = (fp.errors / fp.values) / (standard_fp.errors / standard_fp.values)
        ax.scatter(x, y, marker="_", color=color, label=fit_name)
        sel = np.argsort([x])
        sel = sel[~np.isnan(y[sel])]
        ax.plot(x[sel], y[sel], color=color, alpha=alpha)
    ax.set_ylabel("Relative uncertainty change")
    ax.set_xticks(x)
    ax.set_xticklabels(standard_fp.names, rotation=90)


def plot_external_comparison(table_dir):
    from higgs_couplings_others import coupling_projections

    compare_fits = {}
    for k, error_dict in coupling_projections.items():
        names = list(error_dict.keys())
        values = [_BRS[n] for n in names]
        # coupling_projections is citing relative coupling errors.
        # `* 2` to move to BRs. `* br` to move to total errors.
        errors = [error_dict[n] * 2 * _BRS[n] for n in names]
        compare_fits[k] = alldecays.plotting.util.FitParameters(
            names=names,
            values=values,
            errors=errors,
            covariance=None,
            starting_values=None,
        )

    ds = get_leptonic_ds(table_dir)
    fit = get_fit(ds)
    fits = {"This fit": fit}
    fits.update(compare_fits)
    # fig = alldecays.plotting.compare_values(fits)
    # ax = fig.get_axes()[0]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = alldecays.plotting.compare_errors_only(
        fits,
        ax=ax,
        as_relative_coupling_error=False,  # TODO: Remove?
    )
    ax.set_ylabel("Branching ratio uncertainty")
    add_ild(ax)
    save_hook(fig, "comparison_with_others")


if __name__ == "__main__":

    table_dir = Path(sys.argv[1])
    try:
        n_toys = int(sys.argv[2])
    except (IndexError, ValueError):
        n_toys = 10_000
    plot_channel_for_intro(table_dir)
    plot_matrices(table_dir)
    plot_fits(table_dir)
    plot_toys(table_dir, n_toys=n_toys)
    plot_changed_brs(table_dir, n_toys=n_toys)
    plot_projections(table_dir)
    plot_external_comparison(table_dir)

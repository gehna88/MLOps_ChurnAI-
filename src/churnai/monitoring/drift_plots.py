"""Graphs for the sequential drift study (PNG, matplotlib).

Per step (one iteration of the study):
* ``step<k>_distributions.png`` - the drift features in the initial slice, in all
  training data so far and in the new slice, with PSI and KS in each title
* ``step<k>_psi.png``           - PSI of every feature, against the initial slice and
  against the training data, with the 0.1 / 0.2 bands

Per scenario (the whole study):
* ``psi_over_time.png``, ``psi_heatmap.png``, ``feature_trends.png``,
  ``performance.png``, ``prediction_drift.png``

Across scenarios: ``scenario_comparison.png``.

Colours follow a validated colour-blind-safe categorical order (blue, orange, aqua,
yellow), assigned in that fixed order; text never takes a series colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display: runs in pipelines, CI and the API container

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PLOT_DPI = 130
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4, fixed order
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
CONTEXT_LINE = "#b9b8b3"  # non-focus features: context, not identity
SEQUENTIAL = ("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")
HIST_BINS = 30
LINE_WIDTH = 2.0
MARKER_SIZE = 7
MIN_HEATMAP_WIDTH = 7.0
THRESHOLD_HEADROOM = 1.25  # axis reaches 25% past the drift line
LABEL_GAP = 0.045  # minimum vertical gap between end-of-line labels, as a share of the y range

_STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Opaque, borderless legend box: threshold lines pass behind the text, not through it.
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.facecolor": SURFACE,
    "legend.edgecolor": SURFACE,
    "legend.fontsize": 9,
    "legend.labelcolor": TEXT,
    "font.size": 9,
}


@contextmanager
def _figure(nrows: int = 1, ncols: int = 1, size: tuple[float, float] = (8, 4.5), **kw: Any):
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=size, **kw)
        try:
            yield fig, axes
        finally:
            plt.close(fig)


def _save(fig: plt.Figure, path: Path, title: str, subtitle: str | None = None) -> Path:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    if subtitle:
        fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    return path


def _threshold_lines(ax: plt.Axes, warning: float, alert: float, vertical: bool = False) -> None:
    """Dotted 'moderate' and dashed 'drift' lines; the axis always extends past the drift line,
    so a stable scenario visibly sits far below it."""
    low, high = ax.get_xlim() if vertical else ax.get_ylim()
    (ax.set_xlim if vertical else ax.set_ylim)(low, max(high, alert * THRESHOLD_HEADROOM))
    draw = ax.axvline if vertical else ax.axhline
    for value, label, style in ((warning, f"{warning:g} moderate", ":"), (alert, f"{alert:g} drift", "--")):
        draw(value, color=TEXT_SECONDARY, linestyle=style, linewidth=1.2, zorder=1)
        if vertical:
            ax.text(value, 1.0, f" {label}", transform=ax.get_xaxis_transform(),
                    color=TEXT_SECONDARY, fontsize=8, va="bottom", ha="left")
        else:
            ax.text(1.0, value, f" {label}", transform=ax.get_yaxis_transform(),
                    color=TEXT_SECONDARY, fontsize=8, va="center", ha="left")


def _spread_labels(values: list[float], span: float) -> list[float]:
    """Nudge end-of-line label positions apart so they never overlap."""
    gap = span * LABEL_GAP
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in pairwise(order):
        placed[cur] = max(placed[cur], placed[prev] + gap)
    return placed


def _slice_axis(ax: plt.Axes, slices: Sequence[int], label: str = "New slice (data that arrived)") -> None:
    ax.set_xticks(list(slices))
    ax.set_xticklabels([f"S{s}" for s in slices])
    ax.set_xlabel(label)


# ------------------------------------------------------------- per step ---
def plot_step_distributions(
    initial: pd.DataFrame, training: pd.DataFrame, new: pd.DataFrame, features: list[str],
    psi: dict[str, float], ks: dict[str, dict[str, float]], step: int, new_slice: int,
    scenario: str, path: Path,
) -> Path:
    ncols = 2
    nrows = int(np.ceil(len(features) / ncols))
    with _figure(nrows, ncols, size=(10, 3.4 * nrows), squeeze=False) as (fig, axes):
        groups = (("Initial data (S1)", initial), (f"Trained on (S1-S{new_slice - 1})", training),
                  (f"New data (S{new_slice})", new))
        for ax, feature in zip(axes.flat, features, strict=False):
            pooled = pd.concat([g[feature] for _, g in groups]).astype(float)
            edges = np.histogram_bin_edges(pooled, HIST_BINS)
            for (label, frame), colour in zip(groups, SERIES, strict=False):
                ax.hist(frame[feature].astype(float), bins=edges, density=True, histtype="step",
                        linewidth=LINE_WIDTH, color=colour, label=label)
            ax.set_title(f"{feature}  |  PSI {psi[feature]:.3f}  |  KS {ks[feature]['statistic']:.3f}",
                         loc="left")
            ax.set_ylabel("Density")
            ax.set_yticklabels([])
        for ax in axes.flat[len(features):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.985))
        return _save(fig, path, f"Step {step}: how the new data differs ({scenario})",
                     "PSI and KS compare the new slice with the initial slice S1")


def plot_step_psi(
    psi_initial: dict[str, float], psi_training: dict[str, float], warning: float, alert: float,
    step: int, new_slice: int, scenario: str, path: Path,
) -> Path:
    order = sorted(psi_initial, key=psi_initial.get)
    y = np.arange(len(order))
    height = 0.38
    with _figure(size=(9, 0.42 * len(order) + 1.8)) as (fig, ax):
        ax.barh(y + height / 2, [psi_initial[f] for f in order], height=height - 0.04,
                color=SERIES[0], label="vs initial data (S1)")
        ax.barh(y - height / 2, [psi_training[f] for f in order], height=height - 0.04,
                color=SERIES[1], label=f"vs all training data (S1-S{new_slice - 1})")
        ax.set_yticks(y)
        ax.set_yticklabels(order, color=TEXT)
        ax.set_xlabel("PSI")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(alert * 1.4, max(psi_initial.values()) * 1.12, max(psi_training.values()) * 1.12))
        _threshold_lines(ax, warning, alert, vertical=True)
        ax.legend(loc="lower right")
        return _save(fig, path, f"Step {step}: PSI per feature for new slice S{new_slice} ({scenario})")


# ---------------------------------------------------------- per scenario ---
def plot_psi_over_time(
    steps: list[dict], focus: list[str], warning: float, alert: float, scenario: str, path: Path
) -> Path:
    slices = [s["new_slice"] for s in steps]
    others = [f for f in steps[0]["psi_vs_initial"] if f not in focus]
    with _figure(size=(9, 5)) as (fig, ax):
        for f in others:
            ax.plot(slices, [s["psi_vs_initial"][f] for s in steps], color=CONTEXT_LINE, linewidth=1)
        ax.plot([], [], color=CONTEXT_LINE, linewidth=1, label="Other features")
        ends = []
        for f, colour in zip(focus, SERIES, strict=False):
            values = [s["psi_vs_initial"][f] for s in steps]
            ax.plot(slices, values, color=colour, linewidth=LINE_WIDTH, marker="o",
                    markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=2, label=f)
            ends.append(values[-1])
        top = max(alert * 1.2, *ends, *(s["psi_vs_initial"][f] for s in steps for f in others))
        for f, y in zip(focus, _spread_labels(ends, top), strict=False):
            ax.annotate(f, (slices[-1], y), xytext=(10, 0), textcoords="offset points",
                        color=TEXT, fontsize=9, va="center")
        _threshold_lines(ax, warning, alert)
        _slice_axis(ax, slices)
        ax.set_ylabel("PSI vs initial data (S1)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(slices[0] - 0.3, slices[-1] + 0.8)
        ax.legend(loc="upper left")
        return _save(fig, path, f"Data drift over time ({scenario})",
                     "Each point compares a newly arrived slice with the initial slice S1")


def plot_psi_heatmap(steps: list[dict], warning: float, alert: float, scenario: str, path: Path) -> Path:
    features = sorted(steps[0]["psi_vs_initial"], key=lambda f: -steps[-1]["psi_vs_initial"][f])
    grid = np.array([[s["psi_vs_initial"][f] for s in steps] for f in features])
    cmap = LinearSegmentedColormap.from_list("psi", SEQUENTIAL)
    vmax = max(alert * 2, float(grid.max()))
    size = (max(MIN_HEATMAP_WIDTH, 2.2 + 1.3 * len(steps)), 0.42 * len(features) + 1.8)
    with _figure(size=size) as (fig, ax):
        image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f"S{s['new_slice']}" for s in steps])
        ax.set_xlabel("New slice (data that arrived)")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, color=TEXT)
        for (i, j), value in np.ndenumerate(grid):
            dark = value > vmax * 0.5
            label = f"{value:.2f}" + (" !" if value > alert else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color=SURFACE if dark else TEXT, fontweight="bold" if value > warning else "normal")
        bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label("PSI vs initial data (S1)")
        bar.outline.set_visible(False)
        return _save(fig, path, f"PSI heatmap ({scenario})", f"! marks PSI above {alert:g} (drift)")


def plot_feature_trends(slices: list[dict], focus: list[str], scenario: str, path: Path) -> Path:
    index = [s["index"] for s in slices]
    ncols = 2
    nrows = int(np.ceil(len(focus) / ncols))
    with _figure(nrows, ncols, size=(10, 3.2 * nrows), squeeze=False) as (fig, axes):
        for ax, feature, colour in zip(axes.flat, focus, SERIES, strict=False):
            stats = [s["stats"][feature] for s in slices]
            ax.fill_between(index, [x["p05"] for x in stats], [x["p95"] for x in stats],
                            color=colour, alpha=0.18, linewidth=0, label="5th-95th percentile")
            ax.plot(index, [x["mean"] for x in stats], color=colour, linewidth=LINE_WIDTH, marker="o",
                    markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=2, label="Mean")
            spread = [x["std"] for x in stats]
            ax.set_title(f"{feature}  |  std {spread[0]:,.0f} (S1) to {spread[-1]:,.0f} (S{index[-1]})",
                         loc="left")
            _slice_axis(ax, index, "Slice")
            ax.legend(loc="upper left", fontsize=8)
        for ax in axes.flat[len(focus):]:
            ax.set_visible(False)
        return _save(fig, path, f"Feature trends per slice ({scenario})",
                     "The line is the mean; the band is the 5th-95th percentile, "
                     "so a wider band means more variety")


def plot_performance(steps: list[dict], scenario: str, path: Path) -> Path:
    slices = [s["new_slice"] for s in steps]
    with _figure(1, 2, size=(11, 4.4)) as (fig, axes):
        panels = (("roc_auc", "ROC-AUC", (0.5, 1.0)), ("recall", "Recall (churners caught)", (0.0, 1.0)))
        for ax, (metric, label, limits) in zip(axes, panels, strict=True):
            for key, name, colour in (("retrained_metrics", "Retrained each step", SERIES[0]),
                                      ("stale_metrics", "Never retrained (S1 only)", SERIES[1])):
                ax.plot(slices, [s[key][metric] for s in steps], color=colour, linewidth=LINE_WIDTH,
                        marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=2,
                        label=name)
            ax.set_title(label, loc="left")
            _slice_axis(ax, slices)
            ax.set_ylim(*limits)
        axes[0].legend(loc="lower left")
        return _save(fig, path, f"Model performance on each new slice ({scenario})",
                     "Does retraining on all data so far keep up with drift?")


def plot_prediction_drift(steps: list[dict], warning: float, alert: float, scenario: str, path: Path) -> Path:
    slices = [s["new_slice"] for s in steps]
    with _figure(1, 2, size=(11, 4.4)) as (fig, axes):
        rate_ax, psi_ax = axes
        for key, name, colour in (("actual_churn_rate", "Actual churn rate", SERIES[0]),
                                  ("mean_predicted", "Mean predicted probability", SERIES[1])):
            rate_ax.plot(slices, [s[key] for s in steps], color=colour, linewidth=LINE_WIDTH, marker="o",
                         markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        rate_ax.set_title("Churn: actual vs predicted", loc="left")
        rate_ax.set_ylim(bottom=0)
        rate_ax.legend(loc="upper left")
        _slice_axis(rate_ax, slices)
        psi_ax.bar(slices, [s["prediction_psi"] for s in steps], width=0.5, color=SERIES[0])
        psi_ax.set_title("Prediction drift (PSI of predicted probabilities)", loc="left")
        psi_ax.grid(axis="x", visible=False)
        _threshold_lines(psi_ax, warning, alert)
        _slice_axis(psi_ax, slices)
        return _save(fig, path, f"Prediction drift ({scenario})",
                     "Reference: the model's out-of-fold predictions on its own training data")


# ------------------------------------------------------- across scenarios ---
def plot_scenario_comparison(
    scenarios: list[dict], focus: list[str], warning: float, alert: float, path: Path
) -> Path:
    with _figure(1, 2, size=(11, 4.4)) as (fig, axes):
        psi_ax, auc_ax = axes
        for scenario, colour in zip(scenarios, SERIES, strict=False):
            steps = scenario["steps"]
            slices = [s["new_slice"] for s in steps]
            style = {"color": colour, "linewidth": LINE_WIDTH, "marker": "o", "markersize": MARKER_SIZE,
                     "markeredgecolor": SURFACE, "markeredgewidth": 2, "label": scenario["name"]}
            psi_ax.plot(slices, [max(s["psi_vs_initial"][f] for f in focus) for s in steps], **style)
            auc_ax.plot(slices, [s["retrained_metrics"]["roc_auc"] for s in steps], **style)
            _slice_axis(psi_ax, slices)
            _slice_axis(auc_ax, slices)
        psi_ax.set_title("Largest PSI among the drift features", loc="left")
        psi_ax.set_ylim(bottom=0)
        _threshold_lines(psi_ax, warning, alert)
        auc_ax.set_title("ROC-AUC of the retrained model", loc="left")
        auc_ax.set_ylim(0.5, 1)
        psi_ax.legend(loc="upper left")
        return _save(fig, path, "Scenarios compared",
                     "The natural order is the control: it should stay stable")

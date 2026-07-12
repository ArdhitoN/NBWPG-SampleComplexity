#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

def extract_convergence_data(csv_path):
    """
    Reads the CSV and returns a dict:
      { method_name: [samples_to_converge, ...], ... }
    """
    df = pd.read_csv(csv_path)
    if not {'method_name', 'samples_to_converge'}.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must have 'method_name' and 'samples_to_converge' columns")
    data = {}
    for method, grp in df.groupby('method_name'):
        data[method] = grp['samples_to_converge'].astype(float).tolist()
    return data

def format_method_label(method):
    """
    Formats method names with shortcuts and removes 'sampling' from discounting_free methods.
    """
    clean = method.replace('bias_progression_', '').replace('gain_progression_', '')
    clean = re.sub(r"discounting_free_", r"DF_", clean)
    clean = re.sub(r"discounted_", r"D_", clean)
    clean = re.sub(r"gamma_", r"$\\gamma$=", clean)
    # Match numbers (including decimals) after gamma_ and format to two decimal places
    clean = re.sub(r"(\$\\gamma\$=)(\d+\.\d+)", lambda m: f"{m.group(1)}{float(m.group(2)):.2f}", clean)
    # Apply method name shortcuts
    clean = clean.replace('vanilla', 'V').replace('natural', 'N').replace('proper', 'Prop').replace('popular', 'Pop')
    # Remove 'sampling' from DF_ methods
    clean = clean.replace('_sampling', '')
    return clean

def create_convergence_boxplot(env_name, convergence_data, output_dir):
    """
    Plots a boxplot of samples_to_converge/1000 on log-scale, excluding *_exact methods.
    """
    # Filter out exact methods
    items = [(m, [v/1000 for v in vals]) for m, vals in convergence_data.items() if not m.endswith('_exact')]
    if not items:
        logger.warning("No non-exact methods found, skipping boxplot.")
        return

    methods, samples_lists = zip(*items)
    formatted_labels = [format_method_label(m) for m in methods]

    # Sort methods: D_ before DF_, and V before N/Pop/Prop within each group
    method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
    def custom_sort_key(method):
        prefix = 0 if method.startswith("discounted_") else 1
        method_type = next((mt for mt in method_order.keys() if mt in format_method_label(method)), 'N')
        return (prefix, method_order[method_type], method)
    
    sorted_indices = sorted(range(len(methods)), key=lambda i: custom_sort_key(methods[i]))
    methods = [methods[i] for i in sorted_indices]
    samples_lists = [samples_lists[i] for i in sorted_indices]
    formatted_labels = [formatted_labels[i] for i in sorted_indices]

    # Choose colors by method prefix
    colors = []
    for m in methods:
        if m.startswith("discounting_free_"):
            colors.append("#1f77b4")
        elif m.startswith("discounted_"):
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    # Set font sizes (1.4x for all)
    BASE_TITLE_SIZE = 12  
    TITLE_SIZE = BASE_TITLE_SIZE * 1.4
    BASE_LABEL_SIZE = 10  
    LABEL_SIZE = BASE_LABEL_SIZE * 1.4
    BASE_TICK_SIZE = 10  
    TICK_SIZE = BASE_TICK_SIZE * 1.4

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    bplot = ax.boxplot(
        samples_lists,
        vert=True,
        patch_artist=True,
        labels=formatted_labels,
        widths=0.6
    )

    # Log scale on Y
    ax.set_yscale('log')

    # Color boxes & tick labels
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for lbl, color in zip(ax.get_xticklabels(), colors):
        lbl.set_color(color)
        lbl.set_fontsize(TICK_SIZE)

    # Labels & grid
    ax.set_title(f"Convergence Samples (log-scale, divided by 10³) on {env_name}", fontsize=TITLE_SIZE)
    ax.set_xlabel("Method", fontsize=LABEL_SIZE)
    ax.set_ylabel("Samples to Converge / 10³ (log scale)", fontsize=LABEL_SIZE)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)
    plt.xticks(rotation=0, ha="center")  # Center x-tick labels
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{env_name}_convergence_log_boxplot.png")
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved boxplot to {out_png}")

def create_avg_std_barplot(env_name, convergence_data, output_dir):
    """
    Plots a bar plot with average and std of samples_to_converge on log-scale, excluding *_exact methods,
    with mean and std text below method names as part of x-tick labels, rounded to 0 decimals.
    """
    # Filter out exact methods and compute stats
    items = [(m, np.array(vals)) for m, vals in convergence_data.items() if not m.endswith('_exact')]
    if not items:
        logger.warning("No non-exact methods found, skipping bar plot.")
        return

    methods, samples_arrays = zip(*items)
    formatted_labels = [format_method_label(m) for m in methods]
    means = [np.mean(arr) for arr in samples_arrays]
    stds = [np.std(arr) for arr in samples_arrays]

    # Sort methods: D_ before DF_, and V before N/Pop/Prop within each group
    method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
    def custom_sort_key(method):
        prefix = 0 if method.startswith("discounted_") else 1
        method_type = next((mt for mt in method_order.keys() if mt in format_method_label(method)), 'N')
        return (prefix, method_order[method_type], method)
    
    sorted_indices = sorted(range(len(methods)), key=lambda i: custom_sort_key(methods[i]))
    methods = [methods[i] for i in sorted_indices]
    formatted_labels = [formatted_labels[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]

    # Create extended labels with mean and std below method names
    extended_labels = [f"{label}\nMean: {int(mean)}\nStd: {int(std)}" for label, mean, std in zip(formatted_labels, means, stds)]

    # Choose colors by method prefix
    colors = []
    for m in methods:
        if m.startswith("discounting_free_"):
            colors.append("#1f77b4")
        elif m.startswith("discounted_"):
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    # Set font sizes (1.4x for all)
    BASE_TITLE_SIZE = 12  
    TITLE_SIZE = BASE_TITLE_SIZE * 1.4
    BASE_LABEL_SIZE = 10  
    LABEL_SIZE = BASE_LABEL_SIZE * 1.4
    BASE_TICK_SIZE = 10 
    TICK_SIZE = BASE_TICK_SIZE * 1.4
    BASE_TEXT_SIZE = 8  
    TEXT_SIZE = BASE_TEXT_SIZE * 1.4

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor='black',
        alpha=0.8
    )

    # Set log scale on Y axis
    ax.set_yscale('log')

    # Labels & grid
    ax.set_title(f"Average and Std of Convergence Samples (log-scale) on {env_name}", fontsize=TITLE_SIZE)
    ax.set_xlabel("Method", fontsize=LABEL_SIZE)
    ax.set_ylabel("Samples to Converge (log scale)", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(extended_labels, rotation=0, ha="center", fontsize=TICK_SIZE)  # Center x-tick labels with extended info
    for lbl, color in zip(ax.get_xticklabels(), colors):
        lbl.set_color(color)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{env_name}_convergence_avg_std_barplot.png")
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved bar plot to {out_png}")

def create_convergence_boxplot_no_label(env_name, convergence_data, output_dir):
    """
    Plots a boxplot of samples_to_converge/1000 on log-scale, excluding *_exact methods, without labels.
    """
    # Filter out exact methods
    items = [(m, [v/1000 for v in vals]) for m, vals in convergence_data.items() if not m.endswith('_exact')]
    if not items:
        logger.warning("No non-exact methods found, skipping boxplot.")
        return

    methods, samples_lists = zip(*items)

    # Sort methods: D_ before DF_, and V before N/Pop/Prop within each group
    method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
    def custom_sort_key(method):
        prefix = 0 if method.startswith("discounted_") else 1
        method_type = next((mt for mt in method_order.keys() if mt in format_method_label(method)), 'N')
        return (prefix, method_order[method_type], method)
    
    sorted_indices = sorted(range(len(methods)), key=lambda i: custom_sort_key(methods[i]))
    methods = [methods[i] for i in sorted_indices]
    samples_lists = [samples_lists[i] for i in sorted_indices]

    # Choose colors by method prefix
    colors = []
    for m in methods:
        if m.startswith("discounting_free_"):
            colors.append("#1f77b4")
        elif m.startswith("discounted_"):
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    # Set font sizes (1.4x for all)
    BASE_TITLE_SIZE = 12  
    TITLE_SIZE = BASE_TITLE_SIZE * 1.4
    BASE_TICK_SIZE = 10  
    TICK_SIZE = BASE_TICK_SIZE * 1.4

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    bplot = ax.boxplot(
        samples_lists,
        vert=True,
        patch_artist=True,
        labels=[None] * len(methods),  # Remove method names from x-axis
        widths=0.6
    )

    # Log scale on Y
    ax.set_yscale('log')

    # Color boxes & hide tick labels
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for lbl in ax.get_xticklabels():
        lbl.set_color('white')  # Hide tick labels by setting to background color

    # Labels & grid (no x/y labels)
    ax.set_title(f"Convergence Samples (log-scale, divided by 10³) on {env_name}", fontsize=TITLE_SIZE)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)
    plt.xticks(rotation=0, ha="center")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{env_name}_convergence_log_boxplot_no_labels.png")
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved boxplot to {out_png}")

def create_avg_std_barplot_no_label(env_name, convergence_data, output_dir):
    """
    Plots a bar plot with average and std of samples_to_converge on log-scale, excluding *_exact methods,
    with mean and std text below method names as part of x-tick labels, rounded to 0 decimals, without labels.
    """
    # Filter out exact methods and compute stats
    items = [(m, np.array(vals)) for m, vals in convergence_data.items() if not m.endswith('_exact')]
    if not items:
        logger.warning("No non-exact methods found, skipping bar plot.")
        return

    methods, samples_arrays = zip(*items)
    means = [np.mean(arr) for arr in samples_arrays]
    stds = [np.std(arr) for arr in samples_arrays]

    # Sort methods: D_ before DF_, and V before N/Pop/Prop within each group
    method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
    def custom_sort_key(method):
        prefix = 0 if method.startswith("discounted_") else 1
        method_type = next((mt for mt in method_order.keys() if mt in format_method_label(method)), 'N')
        return (prefix, method_order[method_type], method)
    
    sorted_indices = sorted(range(len(methods)), key=lambda i: custom_sort_key(methods[i]))
    methods = [methods[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]

    # Create extended labels with mean and std below method names
    extended_labels = [f"{format_method_label(m)}\nMean: {int(mean)}\nStd: {int(std)}" for m, mean, std in zip(methods, means, stds)]

    # Choose colors by method prefix
    colors = []
    for m in methods:
        if m.startswith("discounting_free_"):
            colors.append("#1f77b4")
        elif m.startswith("discounted_"):
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    # Set font sizes (1.4x for all)
    BASE_TITLE_SIZE = 12  # Default title size in points
    TITLE_SIZE = BASE_TITLE_SIZE * 1.4
    BASE_TICK_SIZE = 10  # Default tick size in points
    TICK_SIZE = BASE_TICK_SIZE * 1.4
    BASE_TEXT_SIZE = 8  # Default text size for mean/std labels
    TEXT_SIZE = BASE_TEXT_SIZE * 1.4

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor='black',
        alpha=0.8
    )

    # Set log scale on Y axis
    ax.set_yscale('log')

    # Labels & grid (no x/y labels)
    ax.set_title(f"Average and Std of Convergence Samples (log-scale) on {env_name}", fontsize=TITLE_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(extended_labels, rotation=0, ha="center", fontsize=TICK_SIZE)  # Center x-tick labels with extended info
    for lbl, color in zip(ax.get_xticklabels(), colors):
        lbl.set_color(color)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{env_name}_convergence_avg_std_barplot_no_labels.png")
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved bar plot to {out_png}")

def create_avg_std_barplot_log10(env_name, convergence_data, output_dir):
    """
    Bar plot where each bar height = log10(mean samples to converge).
    Y-axis is linear and shows log10 values (e.g. 3, 4, 5, 6).
    Error bars are asymmetric: log10(mean ± std) projected into log10 space.
    X-tick labels still show the raw mean and std for reference.
    """
    items = [(m, np.array(vals)) for m, vals in convergence_data.items() if not m.endswith('_exact')]
    if not items:
        logger.warning("No non-exact methods found, skipping log10 bar plot.")
        return

    methods, samples_arrays = zip(*items)
    formatted_labels = [format_method_label(m) for m in methods]
    means = [np.mean(arr) for arr in samples_arrays]
    stds  = [np.std(arr)  for arr in samples_arrays]

    method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
    def custom_sort_key(method):
        prefix = 0 if method.startswith("discounted_") else 1
        method_type = next((mt for mt in method_order.keys() if mt in format_method_label(method)), 'N')
        return (prefix, method_order[method_type], method)

    sorted_indices = sorted(range(len(methods)), key=lambda i: custom_sort_key(methods[i]))
    methods          = [methods[i]          for i in sorted_indices]
    formatted_labels = [formatted_labels[i] for i in sorted_indices]
    means            = [means[i]            for i in sorted_indices]
    stds             = [stds[i]             for i in sorted_indices]

    log_means   = [np.log10(m) for m in means]
    upper_errs  = [np.log10(m + s) - np.log10(m) for m, s in zip(means, stds)]
    lower_errs  = [np.log10(m) - np.log10(max(m - s, 1)) for m, s in zip(means, stds)]

    extended_labels = [
        f"{label}\nMean: {int(mean)}\nStd: {int(std)}"
        for label, mean, std in zip(formatted_labels, means, stds)
    ]

    colors = []
    for m in methods:
        if m.startswith("discounting_free_"):
            colors.append("#1f77b4")
        elif m.startswith("discounted_"):
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    TITLE_SIZE = 12 * 1.4
    LABEL_SIZE = 10 * 1.4
    TICK_SIZE  = 10 * 1.4

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    x = np.arange(len(methods))
    ax.bar(
        x,
        log_means,
        yerr=[lower_errs, upper_errs],
        capsize=5,
        color=colors,
        edgecolor='black',
        alpha=0.8
    )

    ax.set_title(f"Average and Std of Convergence Samples (log₁₀ values) on {env_name}", fontsize=TITLE_SIZE)
    ax.set_xlabel("Method", fontsize=LABEL_SIZE)
    ax.set_ylabel("log₁₀(Samples to Converge)", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(extended_labels, rotation=0, ha="center", fontsize=TICK_SIZE)
    for lbl, color in zip(ax.get_xticklabels(), colors):
        lbl.set_color(color)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{env_name}_convergence_avg_std_barplot_log10.png")
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved log10 bar plot to {out_png}")


def extract_breakdown_data(csv_path):
    """
    Reads a *_convergence_breakdown.csv and returns:
      { method_name: {'tuning': [..per seed..], 'learning': [..per seed..]} }
    """
    df = pd.read_csv(csv_path)
    required = {'method_name', 'tuning_samples', 'learning_samples'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must have columns {required}")
    data = {}
    for method, grp in df.groupby('method_name'):
        data[method] = {
            'tuning': grp['tuning_samples'].astype(float).tolist(),
            'learning': grp['learning_samples'].astype(float).tolist(),
        }
    return data


def create_stacked_tuning_barplot(env_name, breakdown_data, output_dir):
    """
    Stacked bar plot of mean samples-to-converge split into gamma-tuning vs
    learning (outside tuning), on a log-scale y-axis (consistent with the
    avg/std barplot). Discounted methods show two stacked segments (tuning at
    the bottom, learning on top); discounting-free methods have no tuning so a
    single learning bar.

    breakdown_data: { method_name: {'tuning': [...], 'learning': [...]} }
    """
    items = [(m, d) for m, d in breakdown_data.items() if not m.endswith('_exact')]
    if not items:
        logger.warning("No non-exact methods found, skipping stacked tuning barplot.")
        return

    method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
    def custom_sort_key(method):
        prefix = 0 if method.startswith("discounted_") else 1
        method_type = next((mt for mt in method_order.keys() if mt in format_method_label(method)), 'N')
        return (prefix, method_order[method_type], method)
    items.sort(key=lambda kv: custom_sort_key(kv[0]))

    methods = [m for m, _ in items]
    labels = [format_method_label(m) for m in methods]
    tuning_means = [float(np.mean(d['tuning'])) if d['tuning'] else 0.0 for _, d in items]
    learning_means = [float(np.mean(d['learning'])) if d['learning'] else 0.0 for _, d in items]

    colors = []
    for m in methods:
        if m.startswith("discounting_free_"):
            colors.append("#1f77b4")
        elif m.startswith("discounted_"):
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    TITLE_SIZE = 12 * 1.4
    LABEL_SIZE = 10 * 1.4
    TICK_SIZE = 10 * 1.4

    extended_labels = [
        f"{lbl}\nTune: {int(t)}\nLearn: {int(l)}"
        for lbl, t, l in zip(labels, tuning_means, learning_means)
    ]

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    x = np.arange(len(methods))

    ax.bar(x, tuning_means, color=colors, edgecolor='black', alpha=0.45, hatch='///')
    ax.bar(x, learning_means, bottom=tuning_means, color=colors, edgecolor='black', alpha=0.95)

    ax.set_yscale('log')
    ax.set_title(f"Convergence Samples: Tuning vs Learning (log-scale) on {env_name}", fontsize=TITLE_SIZE)
    ax.set_xlabel("Method", fontsize=LABEL_SIZE)
    ax.set_ylabel("Samples to Converge (log scale)", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(extended_labels, rotation=0, ha="center", fontsize=TICK_SIZE)
    for lbl, color in zip(ax.get_xticklabels(), colors):
        lbl.set_color(color)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)

    legend_elems = [
        Patch(facecolor='gray', alpha=0.45, hatch='///', edgecolor='black', label='Gamma tuning'),
        Patch(facecolor='gray', alpha=0.95, edgecolor='black', label='Learning (outside tuning)'),
    ]
    ax.legend(handles=legend_elems, fontsize=TICK_SIZE, loc='upper right')
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{env_name}_convergence_tuning_stacked_barplot.png")
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved stacked tuning barplot to {out_png}")


def create_stacked_tuning_barplot_log10(env_name, breakdown_data, output_dir):
    """
    Stacked bar plot of samples-to-converge split into gamma-tuning vs learning,
    drawn with log10 VALUES on a linear y-axis (matching the *_barplot_log10 figure:
    ticks read 0,1,2,...,6 and bars start from 0).

    Each bar's total height = log10(total_mean), split at log10(tuning_mean):
      - bottom (hatched)  : tuning, spanning 0 .. log10(tuning_mean)
      - top (solid)       : learning, spanning log10(tuning_mean) .. log10(total_mean)
    Discounting-free methods have tuning=0, so a single learning bar 0 .. log10(total_mean).

    An asymmetric error bar (log10-projected total std) sits on the bar top, and each
    x-label shows the total mean/std plus the tuning/learning means.

    breakdown_data: { method_name: {'tuning': [...], 'learning': [...]} }  (seed-aligned)
    """
    items = [(m, d) for m, d in breakdown_data.items() if not m.endswith('_exact')]
    if not items:
        logger.warning("No non-exact methods found, skipping log10 stacked tuning barplot.")
        return

    method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
    def custom_sort_key(method):
        prefix = 0 if method.startswith("discounted_") else 1
        method_type = next((mt for mt in method_order.keys() if mt in format_method_label(method)), 'N')
        return (prefix, method_order[method_type], method)
    items.sort(key=lambda kv: custom_sort_key(kv[0]))

    methods = [m for m, _ in items]
    labels = [format_method_label(m) for m in methods]
    tuning_means = [float(np.mean(d['tuning'])) if d['tuning'] else 0.0 for _, d in items]
    learning_means = [float(np.mean(d['learning'])) if d['learning'] else 0.0 for _, d in items]
    # Per-seed totals -> total mean/std (tuning is constant, so std comes from learning).
    totals_per_method = [
        [t + l for t, l in zip(d['tuning'], d['learning'])] if d['learning'] else [0.0]
        for _, d in items
    ]
    total_means = [float(np.mean(ts)) for ts in totals_per_method]
    total_stds = [float(np.std(ts)) for ts in totals_per_method]

    # Heights in log10 space.
    log_tuning = [np.log10(t) if t > 0 else 0.0 for t in tuning_means]
    log_total = [np.log10(tot) if tot > 0 else 0.0 for tot in total_means]
    learning_heights = [max(lt - ltun, 0.0) for lt, ltun in zip(log_total, log_tuning)]
    upper_errs = [np.log10(tot + s) - np.log10(tot) if tot > 0 else 0.0
                  for tot, s in zip(total_means, total_stds)]
    lower_errs = [np.log10(tot) - np.log10(max(tot - s, 1)) if tot > 0 else 0.0
                  for tot, s in zip(total_means, total_stds)]

    colors = []
    for m in methods:
        if m.startswith("discounting_free_"):
            colors.append("#1f77b4")
        elif m.startswith("discounted_"):
            colors.append("#ff7f0e")
        else:
            colors.append("#7f7f7f")

    TITLE_SIZE = 12 * 1.4
    LABEL_SIZE = 10 * 1.4
    TICK_SIZE = 10 * 1.4

    extended_labels = [
        f"{lbl}\nMean: {int(tot)}\nStd: {int(s)}\nTune: {int(t)}\nLearn: {int(l)}"
        for lbl, tot, s, t, l in zip(labels, total_means, total_stds, tuning_means, learning_means)
    ]

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    x = np.arange(len(methods))

    # Tuning segment (bottom, hatched). Height 0 for discounting-free methods.
    ax.bar(x, log_tuning, color=colors, edgecolor='black', alpha=0.45, hatch='///')
    # Learning segment (stacked on top, solid), with the total error bar on its top edge.
    ax.bar(x, learning_heights, bottom=log_tuning, color=colors, edgecolor='black', alpha=0.95,
           yerr=[lower_errs, upper_errs], capsize=5, error_kw={'ecolor': 'black'})

    ax.set_title(f"Convergence Samples: Tuning vs Learning (log₁₀ values) on {env_name}", fontsize=TITLE_SIZE)
    ax.set_xlabel("Method", fontsize=LABEL_SIZE)
    ax.set_ylabel("log₁₀(Samples to Converge)", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(extended_labels, rotation=0, ha="center", fontsize=TICK_SIZE)
    for lbl, color in zip(ax.get_xticklabels(), colors):
        lbl.set_color(color)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)

    legend_elems = [
        Patch(facecolor='gray', alpha=0.45, hatch='///', edgecolor='black', label='Gamma tuning'),
        Patch(facecolor='gray', alpha=0.95, edgecolor='black', label='Learning (outside tuning)'),
    ]
    ax.legend(handles=legend_elems, fontsize=TICK_SIZE, loc='upper right')
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{env_name}_convergence_tuning_stacked_barplot_log10.png")
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved log10 stacked tuning barplot to {out_png}")


def export_convergence_stats(convergence_data, output_dir, env_name):
    """
    Exports real and scaled (divided by 1000) average, standard deviation, and median to a CSV.
    """
    stats_data = []
    for method, values in convergence_data.items():
        if not method.endswith('_exact'):
            real_mean = np.mean(values)
            real_std = np.std(values)
            real_median = np.median(values)
            scaled_mean = real_mean / 1000
            scaled_std = real_std / 1000
            scaled_median = real_median / 1000
            stats_data.append({
                'method_name': format_method_label(method),
                'real_mean': real_mean,
                'real_std': real_std,
                'real_median': real_median,
                'scaled_mean': scaled_mean,
                'scaled_std': scaled_std,
                'scaled_median': scaled_median
            })

    if stats_data:
        df = pd.DataFrame(stats_data)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{env_name}_convergence_stats.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved convergence stats to {csv_path}")
    else:
        logger.warning("No non-exact methods found, skipping stats export.")

def main():
    p = argparse.ArgumentParser(
        description="Plot convergence-samples boxplot and avg-std barplot"
    )
    p.add_argument(
        "--csv-path", "-c", default=None,
        help="Path to env-{name}_convergence_data.csv (required unless --breakdown-csv is used)"
    )
    p.add_argument(
        "--breakdown-csv", "-b", default=None,
        help="Path to env-{name}_convergence_breakdown.csv. When given, generates only the "
             "tuning-vs-learning stacked barplot. Additive: touches no other file."
    )
    p.add_argument(
        "--output-dir", "-o", default=".",
        help="Directory to save the plots"
    )
    p.add_argument(
        "--log10-only", action="store_true",
        help="Generate only the log10 avg/std barplot. Does not touch any pre-existing figure or stats file."
    )
    args = p.parse_args()

    # Breakdown mode: additive, generates only the stacked tuning barplot.
    if args.breakdown_csv:
        bname = os.path.basename(args.breakdown_csv)
        env_name = bname.replace("_convergence_breakdown.csv", "") \
            if bname.endswith("_convergence_breakdown.csv") else os.path.splitext(bname)[0]
        try:
            breakdown = extract_breakdown_data(args.breakdown_csv)
        except Exception as e:
            logger.error(f"Failed to extract breakdown data: {e}")
            return
        create_stacked_tuning_barplot(env_name, breakdown, args.output_dir)
        create_stacked_tuning_barplot_log10(env_name, breakdown, args.output_dir)
        return

    if not args.csv_path:
        p.error("--csv-path is required unless --breakdown-csv is given")

    # Infer env name from filename: env-<name>_convergence_data.csv
    fname = os.path.basename(args.csv_path)
    if not fname.endswith("_convergence_data.csv"):
        logger.warning("CSV filename does not follow env-{name}_convergence_data.csv pattern")
        env_name = os.path.splitext(fname)[0]
    else:
        env_name = fname.replace("_convergence_data.csv", "")

    try:
        data = extract_convergence_data(args.csv_path)
    except Exception as e:
        logger.error(f"Failed to extract data: {e}")
        return

    if args.log10_only:
        # Additive only: writes *_convergence_avg_std_barplot_log10.png and nothing else.
        create_avg_std_barplot_log10(env_name, data, args.output_dir)
        return

    create_convergence_boxplot(env_name, data, args.output_dir)
    create_avg_std_barplot(env_name, data, args.output_dir)
    create_avg_std_barplot_log10(env_name, data, args.output_dir)
    create_convergence_boxplot_no_label(env_name, data, args.output_dir)
    create_avg_std_barplot_no_label(env_name, data, args.output_dir)
    export_convergence_stats(data, args.output_dir, env_name)

if __name__ == "__main__":
    main()
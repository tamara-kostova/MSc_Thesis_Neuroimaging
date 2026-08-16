#!/usr/bin/env python3
"""Regenerate the CNN benchmark table and publication figures from a results file.

Reads either the summary CSV of a run or its raw ``benchmark_results_*.json``.
Nothing here retrains or reads a checkpoint -- it is a pure data -> markdown/PNG
rendering step, so the artifacts can be rebuilt without re-executing the
1,800-line training script.

Two runs exist. The 20-epoch one is canonical (it is what the MIPRO paper and the
MSc thesis report); the 5-epoch one is kept for reference only.

    # canonical 20-epoch: table at the top level, figures in 20epoch/
    python results/make_benchmark_artifacts.py \
        results/benchmarks/summary_20260208_214002.csv \
        --out-dir results/benchmarks --only table --epochs 20
    python results/make_benchmark_artifacts.py \
        results/benchmarks/summary_20260208_214002.csv \
        --out-dir results/benchmarks/20epoch --only figures --epochs 20

    # superseded 5-epoch reference figures
    python results/make_benchmark_artifacts.py \
        benchmark_results_5epoch_20260110_212727.json \
        --out-dir results/benchmarks/5epoch --only figures --epochs 5

Metric definitions (see 04_models_training_checkpoint.py:619-629):
    accuracy  plain top-1 accuracy
    f1        F1, average='weighted'
    auc       ROC-AUC; binary on two-class tasks, multi_class='ovr' on multiclass
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Column order and row order are fixed so the regenerated table stays diffable
# against the previously published one.
DATASETS = [
    "MRI_tumor_binary_norm",
    "MRI_tumor_multiclass_norm",
    "MRI_ms_norm",
    "CT_stroke_binary_norm",
]
MODELS = [
    "resnet50",
    "resnet101",
    "vgg16",
    "densenet121",
    "densenet169",
    "mobilenet_v2",
    "efficientnet_b0",
    "efficientnet_b4",
]

# Tasks where an accuracy below chance (0.5) means the artifact is corrupt.
# MRI_ms is two-class as well but is deliberately excluded: its published
# results legitimately straddle chance (46.6-59.7%), which is the finding.
BINARY_TASKS = ["MRI_tumor_binary_norm", "CT_stroke_binary_norm"]

# All-zero placeholder rows from a run that never completed.
DROP_MODELS = ["inception_v3"]

METRICS = ["accuracy", "f1", "auc"]

PARAMS_M = {
    "resnet50": 25.6,
    "resnet101": 44.5,
    "vgg16": 138.4,
    "densenet121": 8.0,
    "densenet169": 14.1,
    "mobilenet_v2": 3.5,
    "efficientnet_b0": 5.3,
    "efficientnet_b4": 19.3,
}


def display_model(name):
    return name.replace("efficientnet_", "EffNet_")


def display_dataset(name):
    return name.replace("_norm", "")


def load_json(json_path):
    """Flatten a raw benchmark_results_*.json into the summary CSV's shape.

    Models that never finished have no 'test_metrics' key at all, so this also
    drops them.
    """
    with open(json_path) as f:
        all_results = json.load(f)

    rows = []
    for dataset, models in all_results.items():
        for model, result in models.items():
            if "test_metrics" not in result:
                print(f"  skipping {dataset}/{model}: no test_metrics (run never completed)")
                continue
            tm = result["test_metrics"]
            rows.append(
                {"dataset": dataset, "model": model, **{k: tm[k] for k in METRICS}}
            )
    return pd.DataFrame(rows)


def load(path):
    """Read a summary CSV or a raw results JSON, and drop placeholder rows."""
    if path.lower().endswith(".json"):
        df = load_json(path)
    else:
        df = pd.read_csv(path)

    missing_cols = {"dataset", "model", *METRICS} - set(df.columns)
    if missing_cols:
        raise SystemExit(f"FATAL: {path} is missing columns: {sorted(missing_cols)}")

    dropped = df[df.model.isin(DROP_MODELS)]
    if len(dropped):
        print(f"  dropping {len(dropped)} placeholder row(s) for {DROP_MODELS}")
        df = df[~df.model.isin(DROP_MODELS)]

    return df.reset_index(drop=True)


def sanity_check(df):
    """Fail loudly before writing anything if the input looks corrupt.

    Each of these would have caught the 0.1% stroke column in the old table.
    """
    errors = []

    for metric in METRICS:
        bad = df[(df[metric] < 0) | (df[metric] > 1)]
        for _, r in bad.iterrows():
            errors.append(f"{r.dataset}/{r.model}: {metric}={r[metric]} outside [0, 1]")

    for _, r in df[df.dataset.isin(BINARY_TASKS)].iterrows():
        if r.accuracy < 0.5:
            errors.append(
                f"{r.dataset}/{r.model}: accuracy={r.accuracy:.4f} is below chance "
                f"(0.5) on a binary task"
            )

    present = set(zip(df.dataset, df.model))
    for dataset in DATASETS:
        for model in MODELS:
            if (dataset, model) not in present:
                errors.append(f"{dataset}/{model}: missing cell")

    dupes = df.groupby(["dataset", "model"]).size()
    for (dataset, model), n in dupes[dupes > 1].items():
        errors.append(f"{dataset}/{model}: {n} duplicate rows")

    if errors:
        print("\nFATAL: sanity check failed -- refusing to write artifacts:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"  sanity check passed ({len(df)} rows, {len(DATASETS)}x{len(MODELS)} cells)")


def markdown_table(df, metric, decimals=1, as_percent=True):
    """One markdown table of `metric`, models as rows and datasets as columns."""
    scale = 100 if as_percent else 1
    suffix = "%" if as_percent else ""

    header = "| Model | " + " | ".join(display_dataset(d) for d in DATASETS) + " |\n"
    header += "|-------|" + "---|" * len(DATASETS) + "\n"

    rows = ""
    for model in MODELS:
        row = f"| {display_model(model)} |"
        for dataset in DATASETS:
            cell = df[(df.model == model) & (df.dataset == dataset)][metric]
            row += (
                f" {cell.values[0] * scale:.{decimals}f}{suffix} |"
                if len(cell)
                else " - |"
            )
        rows += row + "\n"

    return header + rows


def write_table(df, out_dir, source, epochs):
    avg = (
        df.groupby("model")["accuracy"]
        .mean()
        .reindex(MODELS)
        .sort_values(ascending=False)
    )

    parts = [
        f"# CNN benchmark -- final results ({epochs}-epoch run)\n",
        f"Generated from `{os.path.basename(source)}`. Do not hand-edit; "
        "regenerate with `results/make_benchmark_artifacts.py`.\n",
        "## Accuracy\n",
        markdown_table(df, "accuracy"),
        "\n## Weighted F1\n",
        "F1 with `average='weighted'`.\n",
        markdown_table(df, "f1"),
        "\n## AUC\n",
        "ROC-AUC: binary on the two-class tasks, one-vs-rest (`multi_class='ovr'`) "
        "on `MRI_tumor_multiclass`.\n",
        markdown_table(df, "auc", decimals=4, as_percent=False),
        "\n## Average accuracy across the four tasks\n",
        "| Model | Mean accuracy |\n|-------|---|\n",
        "".join(f"| {display_model(m)} | {v * 100:.1f}% |\n" for m, v in avg.items()),
    ]

    table = "\n".join(parts)
    path = os.path.join(out_dir, "FINAL_TABLE.md")
    with open(path, "w") as f:
        f.write(table)
    print(f"  wrote {path}")
    return table


def make_figures(df, out_dir, epochs):
    """The four publication figures.

    The numeric suffix is the epoch count, which is where the original
    `heatmap_20.png` family got its name (they came from the NUM_EPOCHS = 20
    run). Deriving it from --epochs keeps the canonical set on its published
    filenames while making a 5-epoch render land on `_5` instead of silently
    reusing `_20`.
    """
    suffix = f"_{epochs}"
    plot_df = df.copy()
    plot_df["dataset"] = plot_df.dataset.map(display_dataset)
    plot_df["model"] = plot_df.model.map(display_model)
    for metric in METRICS:
        plot_df[metric] = plot_df[metric] * 100

    dataset_order = [display_dataset(d) for d in DATASETS]
    model_order = [display_model(m) for m in MODELS]

    # FIG 1: heatmap (main paper figure)
    pivot = (
        plot_df.pivot(index="model", columns="dataset", values="accuracy")
        .reindex(index=model_order, columns=dataset_order)
        .round(1)
    )
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        pivot,
        annot=True,
        cmap="YlOrRd",
        fmt=".1f",
        vmin=40,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
        linewidths=0.5,
    )
    plt.title(
        f"Neuroimaging classification benchmark "
        f"({len(df)} experiments, {epochs} epochs)",
        fontsize=14,
        pad=20,
    )
    plt.ylabel("Model", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # FIG 2: model ranking, averaged across tasks
    model_avg = plot_df.groupby("model")["accuracy"].mean().sort_values()
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(model_avg)), model_avg.values, color="steelblue")
    plt.yticks(range(len(model_avg)), model_avg.index)
    plt.xlabel("Average accuracy across the four tasks (%)", fontsize=12)
    plt.title(f"Model performance ranking ({epochs} epochs)", fontsize=14)
    plt.xlim(0, 100)
    for i, v in enumerate(model_avg.values):
        plt.text(v + 0.8, i, f"{v:.1f}%", va="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"model_ranking{suffix}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # FIG 3: task difficulty. Chance line, not a fixed 90% line -- it is the only
    # reference that means the same thing on binary and 4-class tasks.
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(
        data=plot_df,
        x="dataset",
        y="accuracy",
        order=dataset_order,
        hue="dataset",
        hue_order=dataset_order,
        palette="Set2",
        legend=False,
    )
    sns.stripplot(
        data=plot_df,
        x="dataset",
        y="accuracy",
        order=dataset_order,
        color="0.25",
        size=4,
        ax=ax,
    )
    plt.title(f"Task difficulty distribution ({epochs} epochs)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=20)
    plt.ylim(40, 102)
    plt.axhline(50, color="crimson", linestyle=":", label="chance (binary tasks)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"task_difficulty{suffix}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # FIG 4: parameter efficiency
    plot_df["params_M"] = plot_df.model.map(
        {display_model(k): v for k, v in PARAMS_M.items()}
    )
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x="params_M",
        y="accuracy",
        hue="dataset",
        style="dataset",
        hue_order=dataset_order,
        style_order=dataset_order,
        s=110,
    )
    plt.xscale("log")
    plt.title(f"Accuracy vs parameter count ({epochs} epochs)")
    plt.xlabel("Parameters (millions, log scale)")
    plt.ylabel("Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Dataset")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"param_efficiency{suffix}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    for stem in ("heatmap", "model_ranking", "task_difficulty", "param_efficiency"):
        name = f"{stem}{suffix}.png"
        print(f"  wrote {os.path.join(out_dir, name)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results", help="summary CSV or raw benchmark_results_*.json for one run"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write artifacts (default: the results file's directory)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="epochs this run trained for; labels the table and figure titles",
    )
    parser.add_argument(
        "--only",
        choices=["all", "table", "figures"],
        default="all",
        help="write only the table, only the figures, or both (default: both)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading {args.results} ({args.epochs}-epoch run)")
    df = load(args.results)
    sanity_check(df)

    table = None
    if args.only in ("all", "table"):
        table = write_table(df, out_dir, args.results, args.epochs)
    if args.only in ("all", "figures"):
        make_figures(df, out_dir, args.epochs)

    if table:
        print("\n" + table)


if __name__ == "__main__":
    main()

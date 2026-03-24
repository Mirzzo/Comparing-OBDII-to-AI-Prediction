from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from maintenance_prediction.experiments import run_training_experiments
from maintenance_prediction.features import build_feature_tables
from maintenance_prediction.modeling import train_and_evaluate


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args[0].startswith("-"):
        raw_args = ["run", *raw_args]

    parser = build_parser()
    args = parser.parse_args(raw_args)

    command = args.command or "run"
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    feature_paths = {
        "train": feature_dir / "train_features.csv",
        "validation": feature_dir / "validation_features.csv",
        "test": feature_dir / "test_features.csv",
    }

    if command == "prepare-features":
        feature_tables = build_and_cache_features(
            dataset_root=dataset_root,
            feature_paths=feature_paths,
            chunksize=args.chunksize,
        )
        print_feature_summary(feature_tables)
        return 0

    if command == "run":
        if args.reuse_features and all(path.exists() for path in feature_paths.values()):
            feature_tables = {
                split: pd.read_csv(path)
                for split, path in feature_paths.items()
            }
            print("Loaded cached feature tables from artifacts/features.")
        else:
            feature_tables = build_and_cache_features(
                dataset_root=dataset_root,
                feature_paths=feature_paths,
                chunksize=args.chunksize,
            )

        print_feature_summary(feature_tables)
        metrics = train_and_evaluate(
            train_frame=feature_tables["train"],
            validation_frame=feature_tables["validation"],
            test_frame=feature_tables["test"],
            output_dir=output_dir,
        )
        print_metrics_summary(metrics)
        return 0

    if command == "run-experiments":
        if args.reuse_features and all(path.exists() for path in feature_paths.values()):
            feature_tables = {
                split: pd.read_csv(path)
                for split, path in feature_paths.items()
            }
            print("Loaded cached feature tables from artifacts/features.")
        else:
            feature_tables = build_and_cache_features(
                dataset_root=dataset_root,
                feature_paths=feature_paths,
                chunksize=args.chunksize,
            )

        print_feature_summary(feature_tables)
        summary = run_training_experiments(
            train_frame=feature_tables["train"],
            validation_frame=feature_tables["validation"],
            test_frame=feature_tables["test"],
            baseline_output_dir=output_dir,
            experiments_output_dir=output_dir / "experiments",
        )
        print_experiment_summary(summary)
        return 0

    parser.error(f"Unsupported command: {command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predictive maintenance experiment runner for the Scania Component X dataset.",
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser(
        "prepare-features",
        help="Build cached train/validation/test feature tables from the raw CSV files.",
    )
    add_shared_arguments(prepare_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Build features when needed, train models, and export evaluation reports.",
    )
    add_shared_arguments(run_parser)
    run_parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse cached feature tables in artifacts/features when they already exist.",
    )

    experiments_parser = subparsers.add_parser(
        "run-experiments",
        help=(
            "Run class-weighted, moderate-oversampling, and binary-risk experiments "
            "without overwriting the main benchmark outputs."
        ),
    )
    add_shared_arguments(experiments_parser)
    experiments_parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse cached feature tables in artifacts/features when they already exist.",
    )

    return parser


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        default="Dataset",
        help="Path to the dataset directory that contains the data/ folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where features, models, and reports will be written.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="CSV chunk size used while streaming operational readouts.",
    )


def build_and_cache_features(
    dataset_root: Path,
    feature_paths: dict[str, Path],
    chunksize: int,
) -> dict[str, pd.DataFrame]:
    print("Building feature tables from raw dataset files...")
    feature_tables = build_feature_tables(dataset_root=dataset_root, chunksize=chunksize)

    for path in feature_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    for split, feature_frame in feature_tables.items():
        feature_frame.to_csv(feature_paths[split], index=False)
        print(f"Saved {split} features to {feature_paths[split]}")

    return feature_tables


def print_feature_summary(feature_tables: dict[str, pd.DataFrame]) -> None:
    for split, feature_frame in feature_tables.items():
        label_counts = (
            feature_frame["class_label"].value_counts().sort_index().to_dict()
            if "class_label" in feature_frame.columns
            else {}
        )
        print(
            f"{split}: rows={len(feature_frame)}, "
            f"columns={len(feature_frame.columns)}, "
            f"class_counts={label_counts}"
        )


def print_metrics_summary(metrics: dict[str, dict[str, object]]) -> None:
    print("Evaluation summary:")
    for model_name, split_metrics in metrics.items():
        validation_metrics = split_metrics["validation"]
        test_metrics = split_metrics["test"]
        print(f"- {model_name}")
        print(f"  validation: {format_metric_line(validation_metrics)}")
        print(f"  test: {format_metric_line(test_metrics)}")


def format_metric_line(metric_block: dict[str, object]) -> str:
    return (
        f"accuracy={metric_block['accuracy']}, "
        f"macro_precision={metric_block['macro_precision']}, "
        f"macro_recall={metric_block['macro_recall']}, "
        f"macro_f1={metric_block['macro_f1']}, "
        f"weighted_f1={metric_block['weighted_f1']}, "
        f"mean_cost={metric_block['challenge_cost_mean']}, "
        f"total_cost={metric_block['challenge_cost_total']}"
    )


def print_experiment_summary(summary: dict[str, object]) -> None:
    print("Experiment summary:")
    print(f"- outputs: {summary['output_dir']}")
    print(f"- markdown summary: {summary['summary_path']}")
    best_cost = summary["best_multiclass_by_test_cost"]
    best_recall = summary["best_multiclass_by_test_macro_recall"]
    best_binary = summary["best_binary_by_test_positive_recall"]
    print(
        "- best multiclass by test mean cost: "
        f"{best_cost.get('model')} ({best_cost.get('value')})"
    )
    print(
        "- best multiclass by test macro recall: "
        f"{best_recall.get('model')} ({best_recall.get('value')})"
    )
    print(
        "- best binary by test positive recall: "
        f"{best_binary.get('model')} ({best_binary.get('value')})"
    )

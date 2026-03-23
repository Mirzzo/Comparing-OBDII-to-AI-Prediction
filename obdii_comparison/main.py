from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    artifacts_dir = project_root / args.artifacts_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = artifacts_dir / "features"
    report_dir = artifacts_dir / "reports"

    validation_features = pd.read_csv(feature_dir / "validation_features.csv")
    test_features = pd.read_csv(feature_dir / "test_features.csv")
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    ai_prediction_tables = load_prediction_tables(report_dir)

    selected_ai_model = args.ai_model or choose_default_ai_model(metrics)
    print(
        "Selected AI model for main comparison: "
        f"{get_display_name(selected_ai_model)}"
    )

    baseline_metrics = metrics["reactive_baseline"]
    baseline_prediction_tables = ai_prediction_tables["reactive_baseline"]
    save_reactive_baseline_outputs(
        output_dir=output_dir,
        baseline_metrics=baseline_metrics,
        baseline_prediction_tables=baseline_prediction_tables,
    )

    comparison_workbook_path = output_dir / "Comparison Table.xlsx"
    workbook_inputs = {
        "metrics": metrics,
        "selected_ai_model": selected_ai_model,
        "class_distribution": build_class_distribution_sheet(
            validation_features=validation_features,
            test_features=test_features,
        ),
        "interpretation_sheet": build_interpretation_sheet(
            metrics=metrics,
            selected_ai_model=selected_ai_model,
            class_distribution=build_class_distribution_sheet(
                validation_features=validation_features,
                test_features=test_features,
            ),
        ),
        "validation_comparison_table": build_comparison_prediction_table(
            ai_model=selected_ai_model,
            ai_prediction_tables=ai_prediction_tables,
            baseline_prediction_frame=baseline_prediction_tables["validation"],
            split="validation",
        ),
        "test_comparison_table": build_comparison_prediction_table(
            ai_model=selected_ai_model,
            ai_prediction_tables=ai_prediction_tables,
            baseline_prediction_frame=baseline_prediction_tables["test"],
            split="test",
        ),
        "validation_baseline_confusion": build_confusion_frame_from_prediction_table(
            baseline_prediction_tables["validation"],
        ),
        "test_baseline_confusion": build_confusion_frame_from_prediction_table(
            baseline_prediction_tables["test"],
        ),
    }
    comparison_workbook_path = write_comparison_workbook_with_fallback(
        workbook_path=comparison_workbook_path,
        **workbook_inputs,
    )

    print("Standalone OBD-II comparison complete.")
    print(f"Workbook written to: {comparison_workbook_path}")
    print(
        "Reactive OBD-II-style baseline vs selected AI model (test split): "
        f"baseline macro_f1={baseline_metrics['test']['macro_f1']}, "
        f"baseline mean_cost={baseline_metrics['test']['challenge_cost_mean']}, "
        f"ai macro_f1={metrics[selected_ai_model]['test']['macro_f1']}, "
        f"ai mean_cost={metrics[selected_ai_model]['test']['challenge_cost_mean']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone OBD-II vs AI comparison workbook from existing "
            "prediction artifacts without changing the training pipeline."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root that contains artifacts/ and Dataset/.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Existing prediction-system artifacts directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="obdii_comparison/artifacts",
        help="Standalone output directory for OBD-II comparison files.",
    )
    parser.add_argument(
        "--ai-model",
        default=None,
        choices=["logistic_regression", "random_forest"],
        help=(
            "AI model to compare directly against the OBD-II proxy. "
            "Defaults to the model with the lowest test challenge cost."
        ),
    )
    return parser


def choose_default_ai_model(metrics: dict[str, dict[str, object]]) -> str:
    return min(
        ["logistic_regression", "random_forest"],
        key=lambda model_name: metrics[model_name]["test"]["challenge_cost_mean"],
    )


def load_prediction_tables(report_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for model_name in ["reactive_baseline", "logistic_regression", "random_forest"]:
        tables[model_name] = {
            "validation": pd.read_csv(report_dir / f"{model_name}_validation_predictions.csv"),
            "test": pd.read_csv(report_dir / f"{model_name}_test_predictions.csv"),
        }
    return tables


def build_comparison_prediction_table(
    ai_model: str,
    ai_prediction_tables: dict[str, dict[str, pd.DataFrame]],
    baseline_prediction_frame: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    ai_prediction_frame = ai_prediction_tables[ai_model][split].rename(
        columns={"predicted_label": "ai_prediction"}
    )
    comparison_frame = baseline_prediction_frame.rename(
        columns={
            "predicted_label": "reactive_obdii_prediction",
        }
    ).merge(
        ai_prediction_frame[["vehicle_id", "ai_prediction"]],
        on="vehicle_id",
        how="left",
    )
    comparison_frame["prediction_agreement"] = (
        comparison_frame["reactive_obdii_prediction"] == comparison_frame["ai_prediction"]
    ).astype(int)
    comparison_frame["reactive_obdii_correct"] = (
        comparison_frame["reactive_obdii_prediction"] == comparison_frame["true_label"]
    ).astype(int)
    comparison_frame["ai_correct"] = (
        comparison_frame["ai_prediction"] == comparison_frame["true_label"]
    ).astype(int)
    comparison_frame["ai_model"] = get_display_name(ai_model)
    return comparison_frame


def write_comparison_workbook(
    workbook_path: Path,
    metrics: dict[str, dict[str, object]],
    selected_ai_model: str,
    class_distribution: pd.DataFrame,
    interpretation_sheet: pd.DataFrame,
    validation_comparison_table: pd.DataFrame,
    test_comparison_table: pd.DataFrame,
    validation_baseline_confusion: pd.DataFrame,
    test_baseline_confusion: pd.DataFrame,
) -> None:
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        build_summary_sheet(metrics, selected_ai_model).to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )
        build_head_to_head_sheet(metrics, selected_ai_model).to_excel(
            writer,
            sheet_name="Baseline vs AI",
            index=False,
        )
        class_distribution.to_excel(
            writer,
            sheet_name="Class Distribution",
            index=False,
        )
        interpretation_sheet.to_excel(
            writer,
            sheet_name="Interpretation",
            index=False,
        )
        validation_comparison_table.to_excel(
            writer,
            sheet_name="Validation Predictions",
            index=False,
        )
        test_comparison_table.to_excel(
            writer,
            sheet_name="Test Predictions",
            index=False,
        )
        validation_baseline_confusion.to_excel(
            writer,
            sheet_name="Validation Baseline CM",
        )
        test_baseline_confusion.to_excel(
            writer,
            sheet_name="Test Baseline CM",
        )


def write_comparison_workbook_with_fallback(
    workbook_path: Path,
    metrics: dict[str, dict[str, object]],
    selected_ai_model: str,
    class_distribution: pd.DataFrame,
    interpretation_sheet: pd.DataFrame,
    validation_comparison_table: pd.DataFrame,
    test_comparison_table: pd.DataFrame,
    validation_baseline_confusion: pd.DataFrame,
    test_baseline_confusion: pd.DataFrame,
) -> Path:
    try:
        write_comparison_workbook(
            workbook_path=workbook_path,
            metrics=metrics,
            selected_ai_model=selected_ai_model,
            class_distribution=class_distribution,
            interpretation_sheet=interpretation_sheet,
            validation_comparison_table=validation_comparison_table,
            test_comparison_table=test_comparison_table,
            validation_baseline_confusion=validation_baseline_confusion,
            test_baseline_confusion=test_baseline_confusion,
        )
        return workbook_path
    except PermissionError:
        fallback_path = workbook_path.with_name("Comparison Table (Refreshed).xlsx")
        write_comparison_workbook(
            workbook_path=fallback_path,
            metrics=metrics,
            selected_ai_model=selected_ai_model,
            class_distribution=class_distribution,
            interpretation_sheet=interpretation_sheet,
            validation_comparison_table=validation_comparison_table,
            test_comparison_table=test_comparison_table,
            validation_baseline_confusion=validation_baseline_confusion,
            test_baseline_confusion=test_baseline_confusion,
        )
        return fallback_path


def build_summary_sheet(
    metrics: dict[str, dict[str, object]],
    selected_ai_model: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_metrics in metrics.items():
        rows.extend(flatten_metric_rows(model_name, model_metrics))

    summary_frame = pd.DataFrame(rows)
    summary_frame["selected_for_main_comparison"] = summary_frame["system_id"].isin(
        {"reactive_baseline", selected_ai_model}
    )
    return summary_frame


def flatten_metric_rows(
    system_name: str,
    split_metrics: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name in ["validation", "test"]:
        metrics = split_metrics[split_name]
        rows.append(
            {
                "system_id": system_name,
                "system": get_display_name(system_name),
                "system_type": get_system_type(system_name),
                "split": split_name,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "challenge_cost_mean": metrics["challenge_cost_mean"],
                "challenge_cost_total": metrics["challenge_cost_total"],
            }
        )
    return rows


def build_head_to_head_sheet(
    metrics: dict[str, dict[str, object]],
    selected_ai_model: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    label_map = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "challenge_cost_mean": "Mean Challenge Cost",
        "challenge_cost_total": "Total Challenge Cost",
    }
    for split_name in ["validation", "test"]:
        for metric_key, metric_label in label_map.items():
            rows.append(
                {
                    "split": split_name,
                    "metric": metric_label,
                    "Reactive OBD-II-style baseline": metrics["reactive_baseline"][split_name][metric_key],
                    get_display_name(selected_ai_model): metrics[selected_ai_model][split_name][metric_key],
                }
            )
    return pd.DataFrame(rows)


def build_class_distribution_sheet(
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, frame in [
        ("validation", validation_features),
        ("test", test_features),
    ]:
        counts = frame["class_label"].value_counts().sort_index()
        total = int(len(frame))
        for class_label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "class_label": int(class_label),
                    "count": int(count),
                    "share": round(float(count / total), 6),
                }
            )
        majority_count = int(counts.max())
        rows.append(
            {
                "split": split_name,
                "class_label": "majority_class_share",
                "count": majority_count,
                "share": round(float(majority_count / total), 6),
            }
        )
    return pd.DataFrame(rows)


def build_interpretation_sheet(
    metrics: dict[str, dict[str, object]],
    selected_ai_model: str,
    class_distribution: pd.DataFrame,
) -> pd.DataFrame:
    logistic_test = metrics["logistic_regression"]["test"]
    random_forest_test = metrics["random_forest"]["test"]
    reactive_test = metrics["reactive_baseline"]["test"]
    validation_majority_share = class_distribution.loc[
        (class_distribution["split"] == "validation")
        & (class_distribution["class_label"] == "majority_class_share"),
        "share",
    ].iloc[0]
    test_majority_share = class_distribution.loc[
        (class_distribution["split"] == "test")
        & (class_distribution["class_label"] == "majority_class_share"),
        "share",
    ].iloc[0]

    logistic_test_report = logistic_test["classification_report"]
    random_forest_test_report = random_forest_test["classification_report"]

    notes = [
        ("Metric consistency", "The workbook now reads all comparison metrics directly from artifacts/reports/metrics.json and the saved prediction CSVs, so the workbook values match the earlier reported results."),
        ("Single baseline name", "The reactive side is labeled consistently as 'Reactive OBD-II-style baseline'. The duplicate 'obdii_proxy' naming has been removed from the workbook."),
        ("Imbalance context", f"Class 0 dominates both evaluation splits: validation majority share = {validation_majority_share:.4f}, test majority share = {test_majority_share:.4f}. This makes accuracy alone misleading."),
        ("Reactive baseline", f"The Reactive OBD-II-style baseline performs poorly on the test split: accuracy = {reactive_test['accuracy']}, macro F1 = {reactive_test['macro_f1']}, mean challenge cost = {reactive_test['challenge_cost_mean']}."),
        ("Random forest interpretation", f"Random Forest has the highest test accuracy ({random_forest_test['accuracy']}) but collapses into the majority class: class 0 recall = {random_forest_test_report['0']['recall']:.4f}, while classes 1-4 recall are all 0.0."),
        ("Why logistic regression", f"Logistic Regression is the main AI comparison because it has the lowest test challenge cost ({logistic_test['challenge_cost_mean']}) and better minority detection than Random Forest."),
        ("Minority detection evidence", f"On the test split, Logistic Regression detects some minority failure windows, including class 1 recall = {logistic_test_report['1']['recall']:.4f} and class 4 recall = {logistic_test_report['4']['recall']:.4f}, while Random Forest gives 0.0 recall for classes 1-4."),
        ("Main comparison model", f"The selected AI model for the main comparison is {get_display_name(selected_ai_model)}."),
    ]
    return pd.DataFrame(notes, columns=["topic", "note"])


def build_confusion_frame_from_prediction_table(
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    labels = [0, 1, 2, 3, 4]
    matrix = pd.crosstab(
        prediction_frame["true_label"],
        prediction_frame["predicted_label"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)
    matrix.index.name = "true_label"
    return matrix


def save_reactive_baseline_outputs(
    output_dir: Path,
    baseline_metrics: dict[str, dict[str, object]],
    baseline_prediction_tables: dict[str, pd.DataFrame],
) -> None:
    (output_dir / "reactive_obdii_baseline_metrics.json").write_text(
        json.dumps(baseline_metrics, indent=2),
        encoding="utf-8",
    )
    baseline_prediction_tables["validation"].to_csv(
        output_dir / "reactive_obdii_baseline_validation_predictions.csv",
        index=False,
    )
    baseline_prediction_tables["test"].to_csv(
        output_dir / "reactive_obdii_baseline_test_predictions.csv",
        index=False,
    )
    build_confusion_frame_from_prediction_table(
        baseline_prediction_tables["validation"],
    ).to_csv(output_dir / "reactive_obdii_baseline_validation_confusion_matrix.csv")
    build_confusion_frame_from_prediction_table(
        baseline_prediction_tables["test"],
    ).to_csv(output_dir / "reactive_obdii_baseline_test_confusion_matrix.csv")


def get_display_name(system_name: str) -> str:
    return {
        "reactive_baseline": "Reactive OBD-II-style baseline",
        "logistic_regression": "Logistic Regression (AI)",
        "random_forest": "Random Forest (AI)",
    }[system_name]


def get_system_type(system_name: str) -> str:
    return {
        "reactive_baseline": "reactive baseline",
        "logistic_regression": "ai model",
        "random_forest": "ai model",
    }[system_name]


if __name__ == "__main__":
    raise SystemExit(main())

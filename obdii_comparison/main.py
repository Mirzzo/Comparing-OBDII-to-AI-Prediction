from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from obdii_comparison.reactive_details import (
    CLASS_TO_REACTIVE_STATUS,
    CLASS_TO_RECOMMENDATION,
    CLASS_TO_RISK_BAND,
    CLASS_TO_WINDOW,
    ReactiveBaselineExplainer,
)

AI_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "catboost",
    "catboost_two_stage",
]


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
    class_distribution = build_class_distribution_sheet(
        validation_features=validation_features,
        test_features=test_features,
    )

    selected_ai_model = args.ai_model or choose_default_ai_model(metrics)
    print(
        "Selected AI model for main comparison: "
        f"{get_display_name(selected_ai_model)}"
    )

    baseline_metrics = metrics["reactive_baseline"]
    baseline_prediction_tables = ai_prediction_tables["reactive_baseline"]
    explainer = ReactiveBaselineExplainer.load(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
    )
    validation_baseline_details = explainer.explain_predictions(
        feature_frame=validation_features,
        prediction_frame=baseline_prediction_tables["validation"],
    )
    test_baseline_details = explainer.explain_predictions(
        feature_frame=test_features,
        prediction_frame=baseline_prediction_tables["test"],
    )
    save_reactive_baseline_outputs(
        output_dir=output_dir,
        baseline_metrics=baseline_metrics,
        baseline_prediction_tables=baseline_prediction_tables,
        validation_baseline_details=validation_baseline_details,
        test_baseline_details=test_baseline_details,
    )

    comparison_workbook_path = output_dir / "Comparison Table.xlsx"
    workbook_inputs = {
        "metrics": metrics,
        "selected_ai_model": selected_ai_model,
        "class_distribution": class_distribution,
        "decision_legend": build_decision_legend_sheet(),
        "interpretation_sheet": build_interpretation_sheet(
            metrics=metrics,
            selected_ai_model=selected_ai_model,
            class_distribution=class_distribution,
        ),
        "validation_comparison_table": build_comparison_prediction_table(
            ai_model=selected_ai_model,
            ai_prediction_tables=ai_prediction_tables,
            baseline_detail_frame=validation_baseline_details,
            split="validation",
        ),
        "test_comparison_table": build_comparison_prediction_table(
            ai_model=selected_ai_model,
            ai_prediction_tables=ai_prediction_tables,
            baseline_detail_frame=test_baseline_details,
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
        choices=AI_MODEL_NAMES,
        help=(
            "AI model to compare directly against the reactive OBD-II-style baseline. "
            "Defaults to the model with the lowest test challenge cost."
        ),
    )
    return parser


def choose_default_ai_model(metrics: dict[str, dict[str, object]]) -> str:
    available_ai_models = [model_name for model_name in AI_MODEL_NAMES if model_name in metrics]
    return min(
        available_ai_models,
        key=lambda model_name: metrics[model_name]["test"]["challenge_cost_mean"],
    )


def load_prediction_tables(
    report_dir: Path,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    requested_model_names = model_names or ["reactive_baseline", *AI_MODEL_NAMES]
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for model_name in requested_model_names:
        tables[model_name] = {
            "validation": pd.read_csv(report_dir / f"{model_name}_validation_predictions.csv"),
            "test": pd.read_csv(report_dir / f"{model_name}_test_predictions.csv"),
        }
    return tables


def build_comparison_prediction_table(
    ai_model: str,
    ai_prediction_tables: dict[str, dict[str, pd.DataFrame]],
    baseline_detail_frame: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    ai_prediction_frame = ai_prediction_tables[ai_model][split].rename(
        columns={"predicted_label": "ai_prediction"}
    )
    comparison_frame = baseline_detail_frame.merge(
        ai_prediction_frame[["vehicle_id", "ai_prediction"]],
        on="vehicle_id",
        how="left",
    )
    comparison_frame["ai_window"] = comparison_frame["ai_prediction"].map(CLASS_TO_WINDOW)
    comparison_frame["ai_risk_band"] = comparison_frame["ai_prediction"].map(CLASS_TO_RISK_BAND)
    comparison_frame["ai_status"] = comparison_frame["ai_prediction"].map(CLASS_TO_REACTIVE_STATUS)
    comparison_frame["ai_recommendation"] = comparison_frame["ai_prediction"].map(
        CLASS_TO_RECOMMENDATION
    )
    comparison_frame["prediction_agreement"] = (
        comparison_frame["reactive_obdii_prediction"] == comparison_frame["ai_prediction"]
    ).map({True: "Yes", False: "No"})
    comparison_frame["reactive_obdii_correct"] = (
        comparison_frame["reactive_obdii_prediction"] == comparison_frame["true_label"]
    ).map({True: "Yes", False: "No"})
    comparison_frame["ai_correct"] = (
        comparison_frame["ai_prediction"] == comparison_frame["true_label"]
    ).map({True: "Yes", False: "No"})
    comparison_frame["ai_model"] = get_display_name(ai_model)
    comparison_frame["comparison_summary"] = comparison_frame.apply(
        lambda row: build_comparison_summary(
            reactive_prediction=int(row["reactive_obdii_prediction"]),
            reactive_rule_band=str(row["rule_risk_band"]),
            bucket_rule_alignment=str(row["bucket_rule_alignment"]),
            ai_prediction=int(row["ai_prediction"]),
            true_label=int(row["true_label"]),
            ai_model_name=get_display_name(ai_model),
        ),
        axis=1,
    )
    ordered_columns = [
        "vehicle_id",
        "true_label",
        "true_window",
        "true_risk_band",
        "reactive_obdii_prediction",
        "reactive_bucket_window",
        "reactive_bucket_risk_band",
        "reactive_bucket_status",
        "reactive_bucket_recommendation",
        "rule_risk_band",
        "rule_status",
        "rule_recommendation",
        "mil_status",
        "pending_issue_count",
        "confirmed_issue_count",
        "severe_issue_count",
        "rule_trigger_summary",
        "reactive_anomaly_score",
        "bucket_rule_alignment",
        "top_triggered_signals",
        "top_triggered_families",
        "reactive_explanation",
        "ai_model",
        "ai_prediction",
        "ai_window",
        "ai_risk_band",
        "ai_status",
        "ai_recommendation",
        "prediction_agreement",
        "reactive_obdii_correct",
        "ai_correct",
        "comparison_summary",
    ]
    return comparison_frame.loc[:, ordered_columns]


def write_comparison_workbook(
    workbook_path: Path,
    metrics: dict[str, dict[str, object]],
    selected_ai_model: str,
    class_distribution: pd.DataFrame,
    decision_legend: pd.DataFrame,
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
        decision_legend.to_excel(
            writer,
            sheet_name="Decision Legend",
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
    decision_legend: pd.DataFrame,
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
            decision_legend=decision_legend,
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
            decision_legend=decision_legend,
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


def build_decision_legend_sheet() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_label in [0, 1, 2, 3, 4]:
        rows.append(
            {
                "class_label": class_label,
                "time_to_failure_window": CLASS_TO_WINDOW[class_label],
                "risk_band": CLASS_TO_RISK_BAND[class_label],
                "reactive_status": CLASS_TO_REACTIVE_STATUS[class_label],
                "recommended_action": CLASS_TO_RECOMMENDATION[class_label],
                "mil_expected": "Usually Off" if class_label < 3 else "Usually On",
            }
        )
    rows.append(
        {
            "class_label": "reactive_bucket",
            "time_to_failure_window": "reactive_bucket_* columns come from the saved reactive baseline prediction class",
            "risk_band": "These bucket fields stay metric-consistent with the original baseline results",
            "reactive_status": "Use them when comparing directly against the AI prediction class",
            "recommended_action": "The bucket is still a proxy, not a real OBD-II ECU output",
            "mil_expected": "It may disagree with the rule-style trigger view on some rows",
        }
    )
    rows.append(
        {
            "class_label": "rule_view",
            "time_to_failure_window": "pending_issue_count = number of last-value signals beyond 3 sigma",
            "risk_band": "confirmed_issue_count = number of signals beyond 4 sigma",
            "reactive_status": "severe_issue_count = number of signals beyond 5 sigma",
            "recommended_action": "top_triggered_signals and top_triggered_families are descriptive proxy fields",
            "mil_expected": "MIL is a rule-style proxy indicator, not a real ECU lamp state",
        }
    )
    return pd.DataFrame(rows)


def build_interpretation_sheet(
    metrics: dict[str, dict[str, object]],
    selected_ai_model: str,
    class_distribution: pd.DataFrame,
) -> pd.DataFrame:
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
    available_ai_models = [model_name for model_name in AI_MODEL_NAMES if model_name in metrics]
    best_accuracy_model = max(
        available_ai_models,
        key=lambda model_name: metrics[model_name]["test"]["accuracy"],
    )
    best_macro_f1_model = max(
        available_ai_models,
        key=lambda model_name: metrics[model_name]["test"]["macro_f1"],
    )
    best_cost_model = min(
        available_ai_models,
        key=lambda model_name: metrics[model_name]["test"]["challenge_cost_mean"],
    )
    best_accuracy_report = metrics[best_accuracy_model]["test"]["classification_report"]
    selected_test = metrics[selected_ai_model]["test"]
    selected_test_report = selected_test["classification_report"]

    notes = [
        ("Metric consistency", "The workbook now reads all comparison metrics directly from artifacts/reports/metrics.json and the saved prediction CSVs, so the workbook values match the earlier reported results."),
        ("Single baseline name", "The reactive side is labeled consistently as 'Reactive OBD-II-style baseline'. The duplicate 'obdii_proxy' naming has been removed from the workbook."),
        ("Imbalance context", f"Class 0 dominates both evaluation splits: validation majority share = {validation_majority_share:.4f}, test majority share = {test_majority_share:.4f}. This makes accuracy alone misleading."),
        ("Reactive baseline", f"The Reactive OBD-II-style baseline performs poorly on the test split: accuracy = {reactive_test['accuracy']}, macro F1 = {reactive_test['macro_f1']}, mean challenge cost = {reactive_test['challenge_cost_mean']}."),
        ("Reactive details", "The prediction sheets now add OBD-style proxy fields such as bucket risk band, rule-level risk band, MIL state, pending/confirmed issue counts, top triggered signals, and a plain-language explanation."),
        ("Bucket vs rule view", "The reactive bucket keeps the original saved class prediction intact, while the rule view restates the same row in a more OBD-style way using threshold counts. If bucket_rule_alignment is 'Mismatch', use the reactive explanation column to interpret that row."),
        ("Highest-accuracy AI model", f"{get_display_name(best_accuracy_model)} has the highest test accuracy ({metrics[best_accuracy_model]['test']['accuracy']}) but should still be read alongside macro metrics because the evaluation splits are heavily imbalanced. Its class 0 recall is {best_accuracy_report['0']['recall']:.4f}."),
        ("Lowest-cost AI model", f"{get_display_name(best_cost_model)} has the lowest test challenge cost ({metrics[best_cost_model]['test']['challenge_cost_mean']}) among the AI models currently available in artifacts/reports."),
        ("Best macro-F1 AI model", f"{get_display_name(best_macro_f1_model)} has the strongest test macro F1 ({metrics[best_macro_f1_model]['test']['macro_f1']}) among the AI models currently available in artifacts/reports."),
        ("Selected AI model behavior", f"{get_display_name(selected_ai_model)} is the main AI comparison model in this workbook. On the test split it has accuracy = {selected_test['accuracy']}, macro precision = {selected_test['macro_precision']}, macro recall = {selected_test['macro_recall']}, macro F1 = {selected_test['macro_f1']}, and mean challenge cost = {selected_test['challenge_cost_mean']}."),
        ("Selected minority detection", f"For the selected AI model, test recall by minority class is class 1 = {selected_test_report['1']['recall']:.4f}, class 2 = {selected_test_report['2']['recall']:.4f}, class 3 = {selected_test_report['3']['recall']:.4f}, and class 4 = {selected_test_report['4']['recall']:.4f}."),
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
    validation_baseline_details: pd.DataFrame,
    test_baseline_details: pd.DataFrame,
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
    validation_baseline_details.to_csv(
        output_dir / "reactive_obdii_baseline_validation_details.csv",
        index=False,
    )
    test_baseline_details.to_csv(
        output_dir / "reactive_obdii_baseline_test_details.csv",
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
        "catboost": "CatBoost (AI)",
        "catboost_two_stage": "Two-Stage CatBoost (AI)",
    }[system_name]


def get_system_type(system_name: str) -> str:
    return {
        "reactive_baseline": "reactive baseline",
        "logistic_regression": "ai model",
        "random_forest": "ai model",
        "catboost": "ai model",
        "catboost_two_stage": "ai model",
    }[system_name]


def build_comparison_summary(
    reactive_prediction: int,
    reactive_rule_band: str,
    bucket_rule_alignment: str,
    ai_prediction: int,
    true_label: int,
    ai_model_name: str,
) -> str:
    reactive_band = CLASS_TO_RISK_BAND[reactive_prediction]
    ai_band = CLASS_TO_RISK_BAND[ai_prediction]

    if reactive_prediction == ai_prediction:
        agreement_text = f"{ai_model_name} and the reactive baseline agree on {ai_band.lower()} risk."
    elif ai_prediction > reactive_prediction:
        agreement_text = f"{ai_model_name} predicts a higher risk band than the reactive baseline."
    else:
        agreement_text = f"The reactive baseline predicts a higher risk band than {ai_model_name}."

    if ai_prediction == true_label and reactive_prediction != true_label:
        correctness_text = "AI matches the true class while the reactive baseline does not."
    elif reactive_prediction == true_label and ai_prediction != true_label:
        correctness_text = "The reactive baseline matches the true class while the AI model does not."
    elif reactive_prediction == true_label and ai_prediction == true_label:
        correctness_text = "Both systems match the true class."
    else:
        correctness_text = "Neither system matches the true class exactly."

    mismatch_text = ""
    if bucket_rule_alignment != "Aligned":
        mismatch_text = (
            f" The reactive bucket is {reactive_band.lower()} but the rule-style trigger view "
            f"reads as {reactive_rule_band.lower()}, so check the explanation column."
        )

    return f"{agreement_text} {correctness_text}{mismatch_text}"


if __name__ == "__main__":
    raise SystemExit(main())

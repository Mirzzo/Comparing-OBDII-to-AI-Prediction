from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from maintenance_prediction.modeling import (
    ESTIMATOR_PREDICT,
    EXPECTED_COST_PREDICT,
    evaluate_predictions,
    generate_predictions,
    infer_feature_groups,
    save_confusion_matrix,
    save_predictions,
    split_features_and_target,
)


@dataclass(frozen=True)
class ExperimentModelConfig:
    estimator: object
    prediction_decoding: str
    oversample_ratio: float | None = None
    class_weight_strategy: str | None = None


def run_training_experiments(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    baseline_output_dir: Path,
    experiments_output_dir: Path,
) -> dict[str, object]:
    experiments_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = baseline_output_dir / "reports" / "metrics.json"
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find baseline metrics at {baseline_metrics_path}. "
            "Run `python main.py run` first so the old results exist for comparison."
        )

    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    (experiments_output_dir / "baseline_metrics_snapshot.json").write_text(
        json.dumps(baseline_metrics, indent=2),
        encoding="utf-8",
    )

    X_train, y_train = split_features_and_target(train_frame)
    X_validation, y_validation = split_features_and_target(validation_frame)
    X_test, y_test = split_features_and_target(test_frame)

    numeric_columns, categorical_columns = infer_feature_groups(X_train)
    original_train_class_counts = y_train.value_counts().sort_index().to_dict()

    class_weight_results = run_multiclass_suite(
        suite_id="exp1_class_weights",
        suite_title="Experiment 1: Class-weighted multiclass models",
        model_configs=build_class_weighted_multiclass_models(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        ),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_frame=validation_frame,
        X_test=X_test,
        y_test=y_test,
        test_frame=test_frame,
        original_train_class_counts=original_train_class_counts,
        experiments_output_dir=experiments_output_dir,
    )

    oversampling_results = run_multiclass_suite(
        suite_id="exp2_moderate_oversampling",
        suite_title="Experiment 2: Moderate random oversampling on train only",
        model_configs=build_moderately_oversampled_multiclass_models(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        ),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_frame=validation_frame,
        X_test=X_test,
        y_test=y_test,
        test_frame=test_frame,
        original_train_class_counts=original_train_class_counts,
        experiments_output_dir=experiments_output_dir,
    )

    binary_results = run_binary_suite(
        suite_id="exp3_binary_risk",
        suite_title="Experiment 3: Binary risk prediction",
        model_configs=build_binary_risk_models(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        ),
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_frame=validation_frame,
        X_test=X_test,
        y_test=y_test,
        test_frame=test_frame,
        original_train_class_counts=original_train_class_counts,
        experiments_output_dir=experiments_output_dir,
        baseline_output_dir=baseline_output_dir,
    )

    multiclass_comparison = build_multiclass_comparison_frame(
        baseline_metrics=baseline_metrics,
        suite_results=[class_weight_results, oversampling_results],
    )
    binary_comparison = build_binary_comparison_frame(
        binary_results=binary_results,
        baseline_output_dir=baseline_output_dir,
    )

    multiclass_comparison.to_csv(
        experiments_output_dir / "multiclass_comparison_vs_old_results.csv",
        index=False,
    )
    binary_comparison.to_csv(
        experiments_output_dir / "binary_risk_comparison_vs_old_multiclass.csv",
        index=False,
    )

    summary_path = experiments_output_dir / "experiment_summary.md"
    summary_path.write_text(
        build_experiment_summary_markdown(
            baseline_metrics=baseline_metrics,
            class_weight_results=class_weight_results,
            oversampling_results=oversampling_results,
            binary_results=binary_results,
            multiclass_comparison=multiclass_comparison,
            binary_comparison=binary_comparison,
        ),
        encoding="utf-8",
    )

    summary = {
        "output_dir": str(experiments_output_dir),
        "summary_path": str(summary_path),
        "multiclass_comparison_path": str(
            experiments_output_dir / "multiclass_comparison_vs_old_results.csv"
        ),
        "binary_comparison_path": str(
            experiments_output_dir / "binary_risk_comparison_vs_old_multiclass.csv"
        ),
        "best_multiclass_by_test_cost": select_best_multiclass_by_metric(
            [class_weight_results, oversampling_results],
            metric_key="challenge_cost_mean",
            lower_is_better=True,
        ),
        "best_multiclass_by_test_macro_recall": select_best_multiclass_by_metric(
            [class_weight_results, oversampling_results],
            metric_key="macro_recall",
            lower_is_better=False,
        ),
        "best_binary_by_test_positive_recall": select_best_binary_by_metric(
            binary_results,
            metric_key="positive_recall",
            lower_is_better=False,
        ),
    }
    (experiments_output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_class_weighted_multiclass_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, ExperimentModelConfig]:
    logistic_preprocessor, tree_preprocessor = build_preprocessors(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    return {
        "logistic_regression_class_weighted": ExperimentModelConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", logistic_preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3_000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            prediction_decoding=EXPECTED_COST_PREDICT,
            class_weight_strategy="balanced",
        ),
        "random_forest_class_weighted": ExperimentModelConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            n_jobs=1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            prediction_decoding=EXPECTED_COST_PREDICT,
            class_weight_strategy="balanced",
        ),
    }


def build_moderately_oversampled_multiclass_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, ExperimentModelConfig]:
    logistic_preprocessor, tree_preprocessor = build_preprocessors(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    return {
        "logistic_regression_oversampled_50pct": ExperimentModelConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", logistic_preprocessor),
                    ("model", LogisticRegression(max_iter=3_000, random_state=42)),
                ]
            ),
            prediction_decoding=EXPECTED_COST_PREDICT,
            oversample_ratio=0.5,
        ),
        "random_forest_oversampled_50pct": ExperimentModelConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            n_jobs=1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            prediction_decoding=EXPECTED_COST_PREDICT,
            oversample_ratio=0.5,
        ),
    }


def build_binary_risk_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, ExperimentModelConfig]:
    logistic_preprocessor, tree_preprocessor = build_preprocessors(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    return {
        "logistic_regression_binary_risk": ExperimentModelConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", logistic_preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3_000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            prediction_decoding=ESTIMATOR_PREDICT,
            class_weight_strategy="balanced",
        ),
        "random_forest_binary_risk": ExperimentModelConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            n_jobs=1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            prediction_decoding=ESTIMATOR_PREDICT,
            class_weight_strategy="balanced",
        ),
    }


def build_preprocessors(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[ColumnTransformer, ColumnTransformer]:
    logistic_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    return logistic_preprocessor, tree_preprocessor


def run_multiclass_suite(
    suite_id: str,
    suite_title: str,
    model_configs: dict[str, ExperimentModelConfig],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_frame: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_frame: pd.DataFrame,
    original_train_class_counts: dict[int, int],
    experiments_output_dir: Path,
) -> dict[str, object]:
    suite_dir = experiments_output_dir / suite_id
    model_dir = suite_dir / "models"
    report_dir = suite_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary: dict[str, object] = {
        "suite_id": suite_id,
        "suite_title": suite_title,
        "task_type": "multiclass",
        "models": {},
    }

    for model_name, config in model_configs.items():
        X_train_model, y_train_model = prepare_training_data(
            X_train=X_train,
            y_train=y_train,
            oversample_ratio=config.oversample_ratio,
        )
        train_class_counts = y_train_model.value_counts().sort_index().to_dict()

        print(f"Training {model_name}...")
        if config.class_weight_strategy is not None:
            print(f"  class weights: {config.class_weight_strategy}")
        if config.oversample_ratio is not None:
            print(f"  oversampling ratio: {config.oversample_ratio}")

        model = config.estimator
        model.fit(X_train_model, y_train_model)
        joblib.dump(model, model_dir / f"{model_name}.joblib")

        validation_predictions = generate_predictions(
            model=model,
            X_frame=X_validation,
            prediction_decoding=config.prediction_decoding,
        )
        test_predictions = generate_predictions(
            model=model,
            X_frame=X_test,
            prediction_decoding=config.prediction_decoding,
        )

        validation_metrics = evaluate_predictions(y_validation, validation_predictions)
        test_metrics = evaluate_predictions(y_test, test_predictions)

        save_predictions(
            frame=validation_frame,
            predictions=validation_predictions,
            output_path=report_dir / f"{model_name}_validation_predictions.csv",
        )
        save_predictions(
            frame=test_frame,
            predictions=test_predictions,
            output_path=report_dir / f"{model_name}_test_predictions.csv",
        )
        save_confusion_matrix(
            y_true=y_validation,
            y_pred=validation_predictions,
            output_path=report_dir / f"{model_name}_validation_confusion_matrix.csv",
        )
        save_confusion_matrix(
            y_true=y_test,
            y_pred=test_predictions,
            output_path=report_dir / f"{model_name}_test_confusion_matrix.csv",
        )

        metrics_summary["models"][model_name] = {
            "training_setup": {
                "original_train_class_counts": original_train_class_counts,
                "train_class_counts_used": train_class_counts,
                "oversample_ratio": config.oversample_ratio,
                "class_weight_strategy": config.class_weight_strategy,
                "prediction_decoding": config.prediction_decoding,
            },
            "validation": validation_metrics,
            "test": test_metrics,
        }

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_summary["models"], indent=2),
        encoding="utf-8",
    )
    return metrics_summary


def run_binary_suite(
    suite_id: str,
    suite_title: str,
    model_configs: dict[str, ExperimentModelConfig],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_frame: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_frame: pd.DataFrame,
    original_train_class_counts: dict[int, int],
    experiments_output_dir: Path,
    baseline_output_dir: Path,
) -> dict[str, object]:
    suite_dir = experiments_output_dir / suite_id
    model_dir = suite_dir / "models"
    report_dir = suite_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    y_train_binary = (y_train != 0).astype(int)
    y_validation_binary = (y_validation != 0).astype(int)
    y_test_binary = (y_test != 0).astype(int)

    metrics_summary: dict[str, object] = {
        "suite_id": suite_id,
        "suite_title": suite_title,
        "task_type": "binary_risk",
        "models": {},
        "baseline_collapsed_binary": load_baseline_binary_metrics(baseline_output_dir),
    }

    for model_name, config in model_configs.items():
        train_counts = y_train_binary.value_counts().sort_index().to_dict()

        print(f"Training {model_name}...")
        if config.class_weight_strategy is not None:
            print(f"  class weights: {config.class_weight_strategy}")

        model = config.estimator
        model.fit(X_train, y_train_binary)
        joblib.dump(model, model_dir / f"{model_name}.joblib")

        validation_predictions = np.asarray(model.predict(X_validation), dtype=int)
        test_predictions = np.asarray(model.predict(X_test), dtype=int)

        validation_metrics = evaluate_binary_predictions(
            y_true=y_validation_binary,
            y_pred=validation_predictions,
        )
        test_metrics = evaluate_binary_predictions(
            y_true=y_test_binary,
            y_pred=test_predictions,
        )

        save_binary_predictions(
            frame=validation_frame,
            true_binary=y_validation_binary,
            predictions=validation_predictions,
            output_path=report_dir / f"{model_name}_validation_predictions.csv",
        )
        save_binary_predictions(
            frame=test_frame,
            true_binary=y_test_binary,
            predictions=test_predictions,
            output_path=report_dir / f"{model_name}_test_predictions.csv",
        )
        save_binary_confusion_matrix(
            y_true=y_validation_binary,
            y_pred=validation_predictions,
            output_path=report_dir / f"{model_name}_validation_confusion_matrix.csv",
        )
        save_binary_confusion_matrix(
            y_true=y_test_binary,
            y_pred=test_predictions,
            output_path=report_dir / f"{model_name}_test_confusion_matrix.csv",
        )

        metrics_summary["models"][model_name] = {
            "training_setup": {
                "original_train_class_counts": original_train_class_counts,
                "binary_train_class_counts_used": train_counts,
                "risk_definition": "positive class = any class_label other than 0",
                "class_weight_strategy": config.class_weight_strategy,
                "prediction_decoding": config.prediction_decoding,
            },
            "validation": validation_metrics,
            "test": test_metrics,
        }

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )
    return metrics_summary


def prepare_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    oversample_ratio: float | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if oversample_ratio is None:
        return X_train, y_train
    return oversample_training_data(X_train, y_train, oversample_ratio)


def oversample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    oversample_ratio: float,
) -> tuple[pd.DataFrame, pd.Series]:
    if oversample_ratio <= 0:
        raise ValueError("oversample_ratio must be positive.")

    training_frame = X_train.copy()
    training_frame["class_label"] = y_train.to_numpy()

    max_class_count = int(training_frame["class_label"].value_counts().max())
    target_count = max(1, int(round(max_class_count * oversample_ratio)))

    resampled_frames: list[pd.DataFrame] = []
    for _, class_frame in training_frame.groupby("class_label", sort=True):
        if len(class_frame) < target_count:
            extra_rows = resample(
                class_frame,
                replace=True,
                n_samples=target_count - len(class_frame),
                random_state=42,
            )
            class_frame = pd.concat([class_frame, extra_rows], ignore_index=True)
        resampled_frames.append(class_frame)

    balanced_frame = (
        pd.concat(resampled_frames, ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    balanced_y = balanced_frame["class_label"].astype(int)
    balanced_X = balanced_frame.drop(columns=["class_label"])
    return balanced_X, balanced_y


def load_baseline_binary_metrics(baseline_output_dir: Path) -> dict[str, dict[str, object]]:
    report_dir = baseline_output_dir / "reports"
    metrics: dict[str, dict[str, object]] = {}
    for model_name in ["logistic_regression", "random_forest"]:
        metrics[model_name] = {}
        for split in ["validation", "test"]:
            prediction_frame = pd.read_csv(report_dir / f"{model_name}_{split}_predictions.csv")
            metrics[model_name][split] = evaluate_binary_predictions(
                y_true=(prediction_frame["true_label"] != 0).astype(int),
                y_pred=(prediction_frame["predicted_label"] != 0).astype(int),
            )
    return metrics


def evaluate_binary_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)

    accuracy = accuracy_score(y_true_array, y_pred_array)
    balanced_accuracy = balanced_accuracy_score(y_true_array, y_pred_array)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            y_true_array,
            y_pred_array,
            labels=[0, 1],
            zero_division=0,
        )
    )
    report = classification_report(
        y_true_array,
        y_pred_array,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "negative_precision": round(float(precision_per_class[0]), 6),
        "negative_recall": round(float(recall_per_class[0]), 6),
        "negative_f1": round(float(f1_per_class[0]), 6),
        "positive_precision": round(float(precision_per_class[1]), 6),
        "positive_recall": round(float(recall_per_class[1]), 6),
        "positive_f1": round(float(f1_per_class[1]), 6),
        "positive_support": int(support_per_class[1]),
        "classification_report": report,
    }


def save_binary_predictions(
    frame: pd.DataFrame,
    true_binary: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray,
    output_path: Path,
) -> None:
    prediction_frame = pd.DataFrame(
        {
            "vehicle_id": frame["vehicle_id"].astype(int),
            "true_risk_label": np.asarray(true_binary, dtype=int),
            "predicted_risk_label": np.asarray(predictions, dtype=int),
        }
    )
    prediction_frame.to_csv(output_path, index=False)


def save_binary_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_frame = pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])
    matrix_frame.index.name = "true_risk_label"
    matrix_frame.to_csv(output_path)


def build_multiclass_comparison_frame(
    baseline_metrics: dict[str, dict[str, object]],
    suite_results: list[dict[str, object]],
) -> pd.DataFrame:
    model_family_map = {
        "logistic_regression_class_weighted": "logistic_regression",
        "random_forest_class_weighted": "random_forest",
        "logistic_regression_oversampled_50pct": "logistic_regression",
        "random_forest_oversampled_50pct": "random_forest",
    }
    comparison_rows: list[dict[str, object]] = []
    metric_keys = ["macro_recall", "macro_f1", "challenge_cost_mean", "accuracy"]

    for suite_result in suite_results:
        for new_model_name, model_metrics in suite_result["models"].items():
            baseline_model_name = model_family_map[new_model_name]
            for split in ["validation", "test"]:
                for metric_key in metric_keys:
                    old_value = baseline_metrics[baseline_model_name][split][metric_key]
                    new_value = model_metrics[split][metric_key]
                    comparison_rows.append(
                        {
                            "suite_id": suite_result["suite_id"],
                            "suite_title": suite_result["suite_title"],
                            "baseline_model": baseline_model_name,
                            "new_model": new_model_name,
                            "split": split,
                            "metric": metric_key,
                            "old_value": old_value,
                            "new_value": new_value,
                            "delta": round(float(new_value) - float(old_value), 6),
                        }
                    )
    return pd.DataFrame(comparison_rows)


def build_binary_comparison_frame(
    binary_results: dict[str, object],
    baseline_output_dir: Path,
) -> pd.DataFrame:
    baseline_binary = load_baseline_binary_metrics(baseline_output_dir)
    model_family_map = {
        "logistic_regression_binary_risk": "logistic_regression",
        "random_forest_binary_risk": "random_forest",
    }
    metric_keys = ["positive_recall", "positive_precision", "positive_f1", "balanced_accuracy"]
    comparison_rows: list[dict[str, object]] = []

    for new_model_name, model_metrics in binary_results["models"].items():
        baseline_model_name = model_family_map[new_model_name]
        for split in ["validation", "test"]:
            for metric_key in metric_keys:
                old_value = baseline_binary[baseline_model_name][split][metric_key]
                new_value = model_metrics[split][metric_key]
                comparison_rows.append(
                    {
                        "suite_id": binary_results["suite_id"],
                        "suite_title": binary_results["suite_title"],
                        "baseline_model": baseline_model_name,
                        "new_model": new_model_name,
                        "split": split,
                        "metric": metric_key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "delta": round(float(new_value) - float(old_value), 6),
                    }
                )
    return pd.DataFrame(comparison_rows)


def select_best_multiclass_by_metric(
    suite_results: list[dict[str, object]],
    metric_key: str,
    lower_is_better: bool,
) -> dict[str, object]:
    best_row: dict[str, object] | None = None
    for suite_result in suite_results:
        for model_name, model_metrics in suite_result["models"].items():
            row = {
                "suite_id": suite_result["suite_id"],
                "suite_title": suite_result["suite_title"],
                "model": model_name,
                "metric": metric_key,
                "value": model_metrics["test"][metric_key],
            }
            if best_row is None:
                best_row = row
                continue
            if lower_is_better:
                if row["value"] < best_row["value"]:
                    best_row = row
            elif row["value"] > best_row["value"]:
                best_row = row
    return best_row or {}


def select_best_binary_by_metric(
    binary_results: dict[str, object],
    metric_key: str,
    lower_is_better: bool,
) -> dict[str, object]:
    best_row: dict[str, object] | None = None
    for model_name, model_metrics in binary_results["models"].items():
        row = {
            "suite_id": binary_results["suite_id"],
            "suite_title": binary_results["suite_title"],
            "model": model_name,
            "metric": metric_key,
            "value": model_metrics["test"][metric_key],
        }
        if best_row is None:
            best_row = row
            continue
        if lower_is_better:
            if row["value"] < best_row["value"]:
                best_row = row
        elif row["value"] > best_row["value"]:
            best_row = row
    return best_row or {}


def build_experiment_summary_markdown(
    baseline_metrics: dict[str, dict[str, object]],
    class_weight_results: dict[str, object],
    oversampling_results: dict[str, object],
    binary_results: dict[str, object],
    multiclass_comparison: pd.DataFrame,
    binary_comparison: pd.DataFrame,
) -> str:
    lines = [
        "# Training Experiments Summary",
        "",
        "These experiments keep the original dataset split and compare new training",
        "strategies against the previously saved multiclass benchmark in",
        "`artifacts/reports/metrics.json`.",
        "",
        "## Old benchmark reference",
        "",
        f"- Logistic Regression test macro recall: {baseline_metrics['logistic_regression']['test']['macro_recall']}",
        f"- Logistic Regression test macro F1: {baseline_metrics['logistic_regression']['test']['macro_f1']}",
        f"- Logistic Regression test mean challenge cost: {baseline_metrics['logistic_regression']['test']['challenge_cost_mean']}",
        f"- Random Forest test macro recall: {baseline_metrics['random_forest']['test']['macro_recall']}",
        f"- Random Forest test macro F1: {baseline_metrics['random_forest']['test']['macro_f1']}",
        f"- Random Forest test mean challenge cost: {baseline_metrics['random_forest']['test']['challenge_cost_mean']}",
        "",
        "## Experiment 1: Class weights",
        "",
    ]
    lines.extend(render_multiclass_suite_markdown(class_weight_results))
    lines.extend(
        [
            "",
            "## Experiment 2: Moderate random oversampling",
            "",
        ]
    )
    lines.extend(render_multiclass_suite_markdown(oversampling_results))
    lines.extend(
        [
            "",
            "## Experiment 3: Binary risk prediction",
            "",
        ]
    )
    lines.extend(render_binary_suite_markdown(binary_results))
    lines.extend(
        [
            "",
            "## Comparison tables",
            "",
            "The CSV comparison files in this folder contain old vs new values and deltas",
            "for the main metrics.",
            "",
            "### Multiclass comparison metrics",
            "",
        ]
    )
    lines.extend(render_comparison_table_markdown(multiclass_comparison))
    lines.extend(
        [
            "",
            "### Binary risk comparison metrics",
            "",
        ]
    )
    lines.extend(render_comparison_table_markdown(binary_comparison))
    return "\n".join(lines) + "\n"


def render_multiclass_suite_markdown(suite_results: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for model_name, metrics in suite_results["models"].items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                f"- Validation macro recall: {metrics['validation']['macro_recall']}",
                f"- Validation macro F1: {metrics['validation']['macro_f1']}",
                f"- Validation mean challenge cost: {metrics['validation']['challenge_cost_mean']}",
                f"- Test macro recall: {metrics['test']['macro_recall']}",
                f"- Test macro F1: {metrics['test']['macro_f1']}",
                f"- Test mean challenge cost: {metrics['test']['challenge_cost_mean']}",
                "",
            ]
        )
    return lines


def render_binary_suite_markdown(binary_results: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for model_name, metrics in binary_results["models"].items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                f"- Validation positive recall: {metrics['validation']['positive_recall']}",
                f"- Validation positive F1: {metrics['validation']['positive_f1']}",
                f"- Validation balanced accuracy: {metrics['validation']['balanced_accuracy']}",
                f"- Test positive recall: {metrics['test']['positive_recall']}",
                f"- Test positive F1: {metrics['test']['positive_f1']}",
                f"- Test balanced accuracy: {metrics['test']['balanced_accuracy']}",
                "",
            ]
        )
    return lines


def render_comparison_table_markdown(comparison_frame: pd.DataFrame) -> list[str]:
    if comparison_frame.empty:
        return ["No comparison rows were generated.", ""]

    display_columns = ["suite_id", "new_model", "split", "metric", "old_value", "new_value", "delta"]
    table = comparison_frame.loc[:, display_columns].copy()
    csv_block = table.to_csv(index=False).strip()
    return ["```csv", csv_block, "```", ""]

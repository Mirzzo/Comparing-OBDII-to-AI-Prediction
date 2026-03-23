from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from maintenance_prediction.baseline import ReactiveThresholdBaseline


CHALLENGE_COST_MATRIX = {
    0: {0: 0, 1: 7, 2: 8, 3: 9, 4: 10},
    1: {0: 200, 1: 0, 2: 7, 3: 8, 4: 9},
    2: {0: 300, 1: 200, 2: 0, 3: 7, 4: 8},
    3: {0: 400, 1: 300, 2: 200, 3: 0, 4: 7},
    4: {0: 500, 1: 400, 2: 300, 3: 200, 4: 0},
}


@dataclass(frozen=True)
class ModelTrainingConfig:
    estimator: object
    oversample_ratio: float | None = None


def train_and_evaluate(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = split_features_and_target(train_frame)
    X_validation, y_validation = split_features_and_target(validation_frame)
    X_test, y_test = split_features_and_target(test_frame)

    numeric_columns, categorical_columns = infer_feature_groups(X_train)
    models = build_models(numeric_columns, categorical_columns)
    original_train_class_counts = y_train.value_counts().sort_index().to_dict()

    metrics_summary: dict[str, dict[str, object]] = {}
    for model_name, training_config in models.items():
        model = training_config.estimator
        X_train_model, y_train_model = prepare_training_data(
            X_train=X_train,
            y_train=y_train,
            oversample_ratio=training_config.oversample_ratio,
        )
        train_class_counts = y_train_model.value_counts().sort_index().to_dict()

        print(f"Training {model_name}...")
        if training_config.oversample_ratio is not None:
            print(
                f"  oversampling applied: ratio={training_config.oversample_ratio}, "
                f"class_counts={train_class_counts}"
            )
        model.fit(X_train_model, y_train_model)
        joblib.dump(model, model_dir / f"{model_name}.joblib")

        validation_predictions = model.predict(X_validation)
        test_predictions = model.predict(X_test)

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

        metrics_summary[model_name] = {
            "training_setup": {
                "original_train_class_counts": original_train_class_counts,
                "oversample_ratio": training_config.oversample_ratio,
                "train_class_counts_used": train_class_counts,
            },
            "validation": validation_metrics,
            "test": test_metrics,
        }

    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    return metrics_summary


def split_features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = frame.drop(columns=["class_label"])
    y = frame["class_label"].astype(int)
    return X, y


def infer_feature_groups(feature_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = [column for column in feature_frame.columns if column.startswith("Spec_")]
    numeric_columns = [
        column
        for column in feature_frame.columns
        if column not in categorical_columns and column != "vehicle_id"
    ]
    return numeric_columns, categorical_columns


def build_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, ModelTrainingConfig]:
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
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
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
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return {
        "reactive_baseline": ModelTrainingConfig(
            estimator=ReactiveThresholdBaseline(),
        ),
        "logistic_regression": ModelTrainingConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", logistic_preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3_000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            oversample_ratio=1.0,
        ),
        "random_forest": ModelTrainingConfig(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            n_jobs=1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    }


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
        raise ValueError("oversample_ratio must be positive when provided.")

    training_frame = X_train.copy()
    training_frame["class_label"] = y_train.to_numpy()

    max_class_count = int(training_frame["class_label"].value_counts().max())
    target_count = max(1, int(round(max_class_count * oversample_ratio)))

    resampled_frames: list[pd.DataFrame] = []
    for class_label, class_frame in training_frame.groupby("class_label", sort=True):
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


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series | list[int]) -> dict[str, object]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    challenge_cost_total, challenge_cost_mean = calculate_challenge_cost(y_true, y_pred)
    return {
        "accuracy": round(float(accuracy), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_precision": round(float(weighted_precision), 6),
        "weighted_recall": round(float(weighted_recall), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "challenge_cost_total": int(challenge_cost_total),
        "challenge_cost_mean": round(float(challenge_cost_mean), 6),
        "classification_report": report,
    }


def save_predictions(
    frame: pd.DataFrame,
    predictions: pd.Series | list[int],
    output_path: Path,
) -> None:
    prediction_frame = pd.DataFrame(
        {
            "vehicle_id": frame["vehicle_id"].astype(int),
            "true_label": frame["class_label"].astype(int),
            "predicted_label": predictions,
        }
    )
    prediction_frame.to_csv(output_path, index=False)


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series | list[int],
    output_path: Path,
) -> None:
    labels = [0, 1, 2, 3, 4]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = "true_label"
    matrix_frame.to_csv(output_path)


def calculate_challenge_cost(
    y_true: pd.Series,
    y_pred: pd.Series | list[int],
) -> tuple[int, float]:
    total_cost = 0
    pair_count = 0
    for actual_label, predicted_label in zip(y_true.tolist(), list(y_pred)):
        total_cost += CHALLENGE_COST_MATRIX[int(actual_label)][int(predicted_label)]
        pair_count += 1

    mean_cost = total_cost / pair_count if pair_count else 0.0
    return total_cost, mean_cost

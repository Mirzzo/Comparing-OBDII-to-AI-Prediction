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
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from maintenance_prediction.baseline import ReactiveThresholdBaseline

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    CatBoostClassifier = None


CHALLENGE_COST_MATRIX = {
    0: {0: 0, 1: 7, 2: 8, 3: 9, 4: 10},
    1: {0: 200, 1: 0, 2: 7, 3: 8, 4: 9},
    2: {0: 300, 1: 200, 2: 0, 3: 7, 4: 8},
    3: {0: 400, 1: 300, 2: 200, 3: 0, 4: 7},
    4: {0: 500, 1: 400, 2: 300, 3: 200, 4: 0},
}
CLASS_LABELS = tuple(sorted(CHALLENGE_COST_MATRIX))
ESTIMATOR_PREDICT = "estimator_predict"
EXPECTED_COST_PREDICT = "expected_cost_minimization"


@dataclass(frozen=True)
class ModelTrainingConfig:
    estimator: object
    prediction_decoding: str = ESTIMATOR_PREDICT
    use_validation_for_fit: bool = False


class TwoStageCatBoostClassifier:
    def __init__(
        self,
        cat_features: list[str],
        random_seed: int = 42,
    ) -> None:
        self.cat_features = cat_features
        self.random_seed = random_seed
        self.fault_threshold_ = 0.5
        self.threshold_selection_strategy_ = (
            "validation_macro_f1_then_mean_cost_then_accuracy"
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        use_best_model: bool = True,
        early_stopping_rounds: int | None = 100,
        verbose: bool | int = False,
    ) -> "TwoStageCatBoostClassifier":
        y_series = pd.Series(y, index=X.index).astype(int)
        binary_y = (y_series != 0).astype(int)
        self.stage1_model_ = self._build_fault_detector()
        stage1_fit_kwargs = self._build_fit_kwargs(
            X=X,
            y=binary_y,
            eval_set=eval_set,
            transform_labels=lambda labels: (pd.Series(labels).astype(int) != 0).astype(int),
            use_best_model=use_best_model,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        self.stage1_model_.fit(X, binary_y, **stage1_fit_kwargs)

        fault_mask = y_series != 0
        fault_X = X.loc[fault_mask]
        fault_y = y_series.loc[fault_mask]
        self.stage2_model_ = self._build_fault_classifier()
        stage2_eval_set: tuple[pd.DataFrame, pd.Series] | None = None
        if eval_set is not None:
            eval_X, eval_y = eval_set
            eval_y_series = pd.Series(eval_y, index=eval_X.index).astype(int)
            eval_fault_mask = eval_y_series != 0
            if eval_fault_mask.any():
                stage2_eval_set = (
                    eval_X.loc[eval_fault_mask],
                    eval_y_series.loc[eval_fault_mask],
                )
        stage2_fit_kwargs = self._build_fit_kwargs(
            X=fault_X,
            y=fault_y,
            eval_set=stage2_eval_set,
            transform_labels=lambda labels: pd.Series(labels).astype(int),
            use_best_model=use_best_model,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        self.stage2_model_.fit(fault_X, fault_y, **stage2_fit_kwargs)

        self.classes_ = np.asarray(CLASS_LABELS, dtype=int)
        if eval_set is not None:
            eval_X, eval_y = eval_set
            self.fault_threshold_ = self._select_fault_threshold(
                X=eval_X,
                y=pd.Series(eval_y, index=eval_X.index).astype(int),
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        fault_probabilities = self._predict_fault_probability(X)
        fault_labels = np.asarray(self.stage2_model_.predict(X), dtype=int).reshape(-1)
        predictions = np.where(fault_probabilities >= self.fault_threshold_, fault_labels, 0)
        return predictions.astype(int)

    def get_training_metadata(self) -> dict[str, object]:
        return {
            "fault_threshold": round(float(self.fault_threshold_), 6),
            "threshold_selection": self.threshold_selection_strategy_,
        }

    def _build_fault_detector(self) -> CatBoostClassifier:
        if CatBoostClassifier is None:
            raise ModuleNotFoundError(
                "catboost is not installed, so the CatBoost experiment variants are unavailable."
            )
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="SqrtBalanced",
            random_seed=self.random_seed,
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
            cat_features=self.cat_features,
        )

    def _build_fault_classifier(self) -> CatBoostClassifier:
        if CatBoostClassifier is None:
            raise ModuleNotFoundError(
                "catboost is not installed, so the CatBoost experiment variants are unavailable."
            )
        return CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="SqrtBalanced",
            random_seed=self.random_seed,
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
            cat_features=self.cat_features,
        )

    def _build_fit_kwargs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None,
        transform_labels,
        use_best_model: bool,
        early_stopping_rounds: int | None,
        verbose: bool | int,
    ) -> dict[str, object]:
        fit_kwargs: dict[str, object] = {}
        if eval_set is not None and len(X) > 0 and len(y) > 0:
            eval_X, eval_y = eval_set
            if len(eval_X) > 0 and len(eval_y) > 0:
                fit_kwargs["eval_set"] = (eval_X, transform_labels(eval_y))
                fit_kwargs["use_best_model"] = use_best_model
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                fit_kwargs["verbose"] = verbose
        return fit_kwargs

    def _predict_fault_probability(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.stage1_model_.predict_proba(X), dtype=float)
        fault_class_index = list(self.stage1_model_.classes_).index(1)
        return probabilities[:, fault_class_index]

    def _select_fault_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        fault_probabilities = self._predict_fault_probability(X)
        fault_labels = np.asarray(self.stage2_model_.predict(X), dtype=int).reshape(-1)
        candidate_thresholds = np.unique(
            np.clip(
                np.concatenate(
                    [
                        np.linspace(0.0, 1.0, 201),
                        fault_probabilities,
                    ]
                ),
                0.0,
                1.0,
            )
        )

        best_threshold = self.fault_threshold_
        best_score: tuple[float, float, float] | None = None
        for threshold in candidate_thresholds:
            predictions = np.where(fault_probabilities >= threshold, fault_labels, 0).astype(int)
            metrics = evaluate_predictions(y, predictions)
            score = (
                float(metrics["macro_f1"]),
                -float(metrics["challenge_cost_mean"]),
                float(metrics["accuracy"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold


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
        train_class_counts = original_train_class_counts

        print(f"Training {model_name}...")
        if training_config.prediction_decoding != ESTIMATOR_PREDICT:
            print(f"  prediction decoding: {training_config.prediction_decoding}")
        fit_kwargs: dict[str, object] = {}
        if training_config.use_validation_for_fit:
            fit_kwargs = {
                "eval_set": (X_validation, y_validation),
                "use_best_model": True,
                "early_stopping_rounds": 100,
                "verbose": False,
            }
            print("  validation-guided early stopping enabled")
        model.fit(X_train, y_train, **fit_kwargs)
        joblib.dump(model, model_dir / f"{model_name}.joblib")

        validation_predictions = generate_predictions(
            model=model,
            X_frame=X_validation,
            prediction_decoding=training_config.prediction_decoding,
        )
        test_predictions = generate_predictions(
            model=model,
            X_frame=X_test,
            prediction_decoding=training_config.prediction_decoding,
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

        metrics_summary[model_name] = {
            "training_setup": {
                "original_train_class_counts": original_train_class_counts,
                "sampling_strategy": "natural",
                "train_class_counts_used": train_class_counts,
                "prediction_decoding": training_config.prediction_decoding,
                "validation_guided_fit": training_config.use_validation_for_fit,
                **extract_model_metadata(model),
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

    models = {
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
            prediction_decoding=EXPECTED_COST_PREDICT,
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
            prediction_decoding=EXPECTED_COST_PREDICT,
        ),
    }
    if CatBoostClassifier is not None:
        models["catboost"] = ModelTrainingConfig(
            estimator=CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                auto_class_weights="SqrtBalanced",
                random_seed=42,
                thread_count=1,
                allow_writing_files=False,
                verbose=False,
                cat_features=categorical_columns,
            ),
            prediction_decoding=ESTIMATOR_PREDICT,
            use_validation_for_fit=True,
        )
        models["catboost_two_stage"] = ModelTrainingConfig(
            estimator=TwoStageCatBoostClassifier(
                cat_features=categorical_columns,
                random_seed=42,
            ),
            prediction_decoding=ESTIMATOR_PREDICT,
            use_validation_for_fit=True,
        )
    return models


def generate_predictions(
    model: object,
    X_frame: pd.DataFrame,
    prediction_decoding: str,
) -> np.ndarray:
    if prediction_decoding == ESTIMATOR_PREDICT:
        return np.asarray(model.predict(X_frame), dtype=int).reshape(-1)
    if prediction_decoding == EXPECTED_COST_PREDICT:
        return predict_with_expected_cost(model, X_frame)
    raise ValueError(f"Unsupported prediction decoding strategy: {prediction_decoding}")


def predict_with_expected_cost(
    model: object,
    X_frame: pd.DataFrame,
) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Expected-cost decoding requires a model with predict_proba().")

    probabilities = np.asarray(model.predict_proba(X_frame), dtype=float)
    actual_classes = [int(label) for label in model.classes_]
    prediction_labels = np.asarray(CLASS_LABELS, dtype=int)
    cost_matrix = np.asarray(
        [
            [CHALLENGE_COST_MATRIX[actual_label][predicted_label] for predicted_label in prediction_labels]
            for actual_label in actual_classes
        ],
        dtype=float,
    )
    expected_cost = probabilities @ cost_matrix
    return prediction_labels[expected_cost.argmin(axis=1)]


def extract_model_metadata(model: object) -> dict[str, object]:
    if hasattr(model, "get_training_metadata"):
        return dict(model.get_training_metadata())
    return {}


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

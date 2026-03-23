from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProxyPredictionResult:
    predicted_labels: np.ndarray
    anomaly_scores: np.ndarray
    dtc_like_counts: np.ndarray
    mil_on_flags: np.ndarray


class OBDIIProxyBaseline:
    """Standalone OBD-II-style proxy detector for comparison reporting.

    This baseline is intentionally separate from the AI training pipeline.
    It uses percentile thresholds learned from healthy class-0 training rows and
    produces DTC-like anomaly counts from the engineered feature tables.
    """

    def __init__(
        self,
        lower_quantile_last: float = 0.005,
        upper_quantile_last: float = 0.995,
        lower_quantile_delta: float = 0.01,
        upper_quantile_delta: float = 0.99,
        delta_weight: float = 0.5,
    ) -> None:
        self.lower_quantile_last = lower_quantile_last
        self.upper_quantile_last = upper_quantile_last
        self.lower_quantile_delta = lower_quantile_delta
        self.upper_quantile_delta = upper_quantile_delta
        self.delta_weight = delta_weight

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OBDIIProxyBaseline":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OBDIIProxyBaseline expects a pandas DataFrame.")

        y_series = pd.Series(y).astype(int)
        self.last_feature_columns_ = [
            column for column in X.columns if column.endswith("_last")
        ]
        self.delta_feature_columns_ = [
            column for column in X.columns if column.endswith("_delta")
        ]
        if not self.last_feature_columns_:
            raise ValueError("No '_last' feature columns were found.")

        numeric_last = self._prepare_numeric_frame(X, self.last_feature_columns_)
        numeric_delta = self._prepare_numeric_frame(X, self.delta_feature_columns_)
        healthy_mask = y_series == 0

        healthy_last = numeric_last.loc[healthy_mask]
        if healthy_last.empty:
            healthy_last = numeric_last
        self.last_lower_bounds_ = healthy_last.quantile(self.lower_quantile_last)
        self.last_upper_bounds_ = healthy_last.quantile(self.upper_quantile_last)

        if self.delta_feature_columns_:
            healthy_delta = numeric_delta.loc[healthy_mask]
            if healthy_delta.empty:
                healthy_delta = numeric_delta
            self.delta_lower_bounds_ = healthy_delta.quantile(self.lower_quantile_delta)
            self.delta_upper_bounds_ = healthy_delta.quantile(self.upper_quantile_delta)
        else:
            self.delta_lower_bounds_ = pd.Series(dtype=float)
            self.delta_upper_bounds_ = pd.Series(dtype=float)

        anomaly_scores, _, _ = self._calculate_anomaly_outputs(X)

        class_centroids = []
        for class_id in sorted(y_series.unique()):
            class_scores = anomaly_scores[y_series.to_numpy() == class_id]
            if len(class_scores) == 0:
                continue
            class_centroids.append((class_id, float(np.nanmedian(class_scores))))

        self.classes_ = np.array([class_id for class_id, _ in class_centroids], dtype=int)
        centroid_values = np.array([value for _, value in class_centroids], dtype=float)
        self.boundaries_ = np.array(
            [(left + right) / 2.0 for left, right in zip(centroid_values[:-1], centroid_values[1:])],
            dtype=float,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_with_details(X).predicted_labels

    def predict_with_details(self, X: pd.DataFrame) -> ProxyPredictionResult:
        if not hasattr(self, "last_feature_columns_"):
            raise ValueError("The OBD-II proxy baseline must be fitted before prediction.")

        anomaly_scores, dtc_like_counts, mil_on_flags = self._calculate_anomaly_outputs(X)
        class_indexes = np.digitize(anomaly_scores, self.boundaries_, right=False)
        predicted_labels = self.classes_[class_indexes]
        return ProxyPredictionResult(
            predicted_labels=predicted_labels,
            anomaly_scores=anomaly_scores,
            dtc_like_counts=dtc_like_counts,
            mil_on_flags=mil_on_flags,
        )

    def _calculate_anomaly_outputs(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        numeric_last = self._prepare_numeric_frame(X, self.last_feature_columns_)
        last_flags = (
            numeric_last.lt(self.last_lower_bounds_, axis=1)
            | numeric_last.gt(self.last_upper_bounds_, axis=1)
        )
        dtc_like_counts = last_flags.sum(axis=1).to_numpy(dtype=int)

        delta_score = np.zeros(len(X), dtype=float)
        if self.delta_feature_columns_:
            numeric_delta = self._prepare_numeric_frame(X, self.delta_feature_columns_)
            delta_flags = (
                numeric_delta.lt(self.delta_lower_bounds_, axis=1)
                | numeric_delta.gt(self.delta_upper_bounds_, axis=1)
            )
            delta_score = delta_flags.sum(axis=1).to_numpy(dtype=float) * self.delta_weight

        anomaly_scores = dtc_like_counts.astype(float) + delta_score
        mil_on_flags = (dtc_like_counts > 0).astype(int)
        return anomaly_scores, dtc_like_counts, mil_on_flags

    def _prepare_numeric_frame(
        self,
        X: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        if not columns:
            return pd.DataFrame(index=X.index)
        missing_columns = [column for column in columns if column not in X.columns]
        if missing_columns:
            raise ValueError(
                "Input data is missing OBD-II proxy features: "
                + ", ".join(sorted(missing_columns)[:10])
            )
        return X.loc[:, columns].apply(pd.to_numeric, errors="coerce")

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ReactiveThresholdBaseline(BaseEstimator, ClassifierMixin):
    """A simple OBD-style reactive detector based on threshold exceedance.

    The baseline learns a "healthy" reference distribution from class 0 samples
    and scores each observation by counting how far its current readout drifts
    outside a normal z-score band. Those anomaly scores are then mapped into the
    dataset's five time-to-failure classes using score boundaries learned from
    the training set.
    """

    def __init__(self, last_feature_suffix: str = "_last", normal_z: float = 3.0):
        self.last_feature_suffix = last_feature_suffix
        self.normal_z = normal_z

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReactiveThresholdBaseline":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ReactiveThresholdBaseline expects a pandas DataFrame.")

        y_series = pd.Series(y).astype(int)
        self.feature_columns_ = [
            column for column in X.columns if column.endswith(self.last_feature_suffix)
        ]
        if not self.feature_columns_:
            raise ValueError("No last-readout features were found for the baseline.")

        numeric_frame = self._prepare_numeric_frame(X, self.feature_columns_)
        reference_frame = numeric_frame.loc[y_series == 0]
        if reference_frame.empty:
            reference_frame = numeric_frame

        self.reference_mean_ = reference_frame.mean(axis=0).to_numpy(dtype=float)
        reference_std = reference_frame.std(axis=0).replace(0.0, np.nan).fillna(1.0)
        self.reference_std_ = reference_std.to_numpy(dtype=float)

        scores = self._score_numeric_frame(numeric_frame)
        class_centroids = []
        for class_id in sorted(y_series.unique()):
            class_scores = scores[y_series.to_numpy() == class_id]
            class_centroids.append((class_id, float(np.nanmedian(class_scores))))

        self.classes_ = np.array([class_id for class_id, _ in class_centroids], dtype=int)
        centroid_values = np.array(
            [score for _, score in class_centroids],
            dtype=float,
        )
        self.boundaries_ = np.array(
            [
                (left + right) / 2.0
                for left, right in zip(centroid_values[:-1], centroid_values[1:])
            ],
            dtype=float,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "feature_columns_"):
            raise ValueError("The baseline must be fitted before calling predict().")

        numeric_frame = self._prepare_numeric_frame(X, self.feature_columns_)
        scores = self._score_numeric_frame(numeric_frame)
        class_indexes = np.digitize(scores, self.boundaries_, right=False)
        return self.classes_[class_indexes]

    def _prepare_numeric_frame(
        self,
        X: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        missing_columns = [column for column in feature_columns if column not in X.columns]
        if missing_columns:
            raise ValueError(
                "Input data is missing baseline features: "
                + ", ".join(sorted(missing_columns)[:10])
            )

        return X.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")

    def _score_numeric_frame(self, numeric_frame: pd.DataFrame) -> np.ndarray:
        values = numeric_frame.to_numpy(dtype=float, copy=False)
        safe_std = np.clip(self.reference_std_, 1e-6, None)
        z_scores = np.abs(values - self.reference_mean_) / safe_std
        exceedance = np.clip(z_scores - self.normal_z, 0.0, None)
        return np.nansum(exceedance, axis=1)


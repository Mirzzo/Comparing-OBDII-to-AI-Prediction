from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


CLASS_TARGET_DELTAS = {
    0: 72.0,
    1: 36.0,
    2: 18.0,
    3: 9.0,
    4: 3.0,
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def train_operational(self) -> Path:
        return self.data_dir / "train_operational_readouts.csv"

    @property
    def validation_operational(self) -> Path:
        return self.data_dir / "validation_operational_readouts.csv"

    @property
    def test_operational(self) -> Path:
        return self.data_dir / "test_operational_readouts.csv"

    @property
    def train_specs(self) -> Path:
        return self.data_dir / "train_specifications.csv"

    @property
    def validation_specs(self) -> Path:
        return self.data_dir / "validation_specifications.csv"

    @property
    def test_specs(self) -> Path:
        return self.data_dir / "test_specifications.csv"

    @property
    def train_tte(self) -> Path:
        return self.data_dir / "train_tte.csv"

    @property
    def validation_labels(self) -> Path:
        return self.data_dir / "validation_labels.csv"

    @property
    def test_labels(self) -> Path:
        return self.data_dir / "test_labels.csv"


def build_feature_tables(
    dataset_root: Path,
    chunksize: int = 50_000,
) -> dict[str, pd.DataFrame]:
    dataset_paths = DatasetPaths(dataset_root)
    train_features = build_training_features(dataset_paths, chunksize)
    validation_features = build_holdout_features(dataset_paths, "validation", chunksize)
    test_features = build_holdout_features(dataset_paths, "test", chunksize)
    return {
        "train": train_features,
        "validation": validation_features,
        "test": test_features,
    }


def build_training_features(
    dataset_paths: DatasetPaths,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    specs = pd.read_csv(dataset_paths.train_specs).set_index("vehicle_id")
    tte = pd.read_csv(dataset_paths.train_tte).set_index("vehicle_id")
    sensor_columns = infer_sensor_columns(dataset_paths.train_operational)

    feature_rows: list[dict[str, object]] = []
    for vehicle_id, vehicle_frame in iter_vehicle_readouts(
        dataset_paths.train_operational,
        chunksize=chunksize,
    ):
        if vehicle_id not in specs.index or vehicle_id not in tte.index:
            continue

        spec_row = get_vehicle_row(specs, vehicle_id)
        tte_row = get_vehicle_row(tte, vehicle_id)
        time_steps = vehicle_frame["time_step"].to_numpy(dtype=float, copy=False)
        selected_snapshots = select_training_snapshots(
            time_steps=time_steps,
            length_of_study_time_step=float(tte_row["length_of_study_time_step"]),
            in_study_repair=int(tte_row["in_study_repair"]),
        )
        if not selected_snapshots:
            continue

        for cutoff_index, class_label in selected_snapshots:
            feature_row = summarize_vehicle_snapshot(
                vehicle_frame=vehicle_frame,
                sensor_columns=sensor_columns,
                cutoff_index=cutoff_index,
                spec_row=spec_row,
            )
            feature_row["class_label"] = int(class_label)
            feature_rows.append(feature_row)

    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        raise ValueError("No training features were generated from the dataset.")
    return optimize_feature_types(feature_frame)


def build_holdout_features(
    dataset_paths: DatasetPaths,
    split: str,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    if split not in {"validation", "test"}:
        raise ValueError("split must be either 'validation' or 'test'.")

    operational_path = (
        dataset_paths.validation_operational
        if split == "validation"
        else dataset_paths.test_operational
    )
    specs_path = dataset_paths.validation_specs if split == "validation" else dataset_paths.test_specs
    labels_path = dataset_paths.validation_labels if split == "validation" else dataset_paths.test_labels

    specs = pd.read_csv(specs_path).set_index("vehicle_id")
    labels = pd.read_csv(labels_path).set_index("vehicle_id")
    sensor_columns = infer_sensor_columns(operational_path)

    feature_rows: list[dict[str, object]] = []
    for vehicle_id, vehicle_frame in iter_vehicle_readouts(operational_path, chunksize=chunksize):
        if vehicle_id not in specs.index or vehicle_id not in labels.index:
            continue

        spec_row = get_vehicle_row(specs, vehicle_id)
        label_row = get_vehicle_row(labels, vehicle_id)
        feature_row = summarize_vehicle_snapshot(
            vehicle_frame=vehicle_frame,
            sensor_columns=sensor_columns,
            cutoff_index=len(vehicle_frame) - 1,
            spec_row=spec_row,
        )
        feature_row["class_label"] = int(label_row["class_label"])
        feature_rows.append(feature_row)

    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        raise ValueError(f"No {split} features were generated from the dataset.")
    return optimize_feature_types(feature_frame)


def infer_sensor_columns(csv_path: Path) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0)
    return [column for column in header.columns if column not in {"vehicle_id", "time_step"}]


def iter_vehicle_readouts(
    csv_path: Path,
    chunksize: int = 50_000,
) -> Iterator[tuple[int, pd.DataFrame]]:
    carryover: pd.DataFrame | None = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if carryover is not None and not carryover.empty:
            chunk = pd.concat([carryover, chunk], ignore_index=True)

        chunk = chunk.sort_values(["vehicle_id", "time_step"], kind="mergesort")
        last_vehicle_id = int(chunk["vehicle_id"].iloc[-1])
        carryover = chunk.loc[chunk["vehicle_id"] == last_vehicle_id].copy()
        complete_rows = chunk.loc[chunk["vehicle_id"] != last_vehicle_id]

        for vehicle_id, vehicle_frame in complete_rows.groupby("vehicle_id", sort=False):
            yield int(vehicle_id), vehicle_frame.reset_index(drop=True)

    if carryover is not None and not carryover.empty:
        carryover = carryover.sort_values(["vehicle_id", "time_step"], kind="mergesort")
        for vehicle_id, vehicle_frame in carryover.groupby("vehicle_id", sort=False):
            yield int(vehicle_id), vehicle_frame.reset_index(drop=True)


def get_vehicle_row(frame: pd.DataFrame, vehicle_id: int) -> pd.Series:
    row = frame.loc[vehicle_id]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def select_training_snapshots(
    time_steps: np.ndarray,
    length_of_study_time_step: float,
    in_study_repair: int,
) -> list[tuple[int, int]]:
    deltas = length_of_study_time_step - time_steps
    selected_snapshots: list[tuple[int, int]] = []

    target_labels = [0, 1, 2, 3, 4] if in_study_repair == 1 else [0]
    for class_label in target_labels:
        mask = build_class_mask(deltas, class_label)
        if not np.any(mask):
            continue

        candidate_indexes = np.flatnonzero(mask)
        target_delta = CLASS_TARGET_DELTAS[class_label]
        chosen_index = candidate_indexes[
            np.argmin(np.abs(deltas[candidate_indexes] - target_delta))
        ]
        selected_snapshots.append((int(chosen_index), int(class_label)))

    return selected_snapshots


def build_class_mask(deltas: np.ndarray, class_label: int) -> np.ndarray:
    if class_label == 4:
        return (deltas >= 0.0) & (deltas <= 6.0)
    if class_label == 3:
        return (deltas > 6.0) & (deltas <= 12.0)
    if class_label == 2:
        return (deltas > 12.0) & (deltas <= 24.0)
    if class_label == 1:
        return (deltas > 24.0) & (deltas <= 48.0)
    if class_label == 0:
        return deltas > 48.0
    raise ValueError(f"Unsupported class label: {class_label}")


def summarize_vehicle_snapshot(
    vehicle_frame: pd.DataFrame,
    sensor_columns: list[str],
    cutoff_index: int,
    spec_row: pd.Series,
) -> dict[str, object]:
    snapshot = vehicle_frame.iloc[: cutoff_index + 1]
    time_steps = snapshot["time_step"].to_numpy(dtype=float, copy=False)
    sensor_values = snapshot.loc[:, sensor_columns].to_numpy(dtype=float, copy=False)

    first_values = sensor_values[0]
    last_values = sensor_values[-1]
    mean_values, std_values = compute_snapshot_statistics(sensor_values)
    delta_values = last_values - first_values

    feature_row: dict[str, object] = {
        "vehicle_id": int(snapshot["vehicle_id"].iloc[0]),
        "history_length": int(len(snapshot)),
        "first_time_step": float(time_steps[0]),
        "last_time_step": float(time_steps[-1]),
        "time_step_span": float(time_steps[-1] - time_steps[0]),
        "mean_step_interval": float(np.mean(np.diff(time_steps))) if len(time_steps) > 1 else 0.0,
    }

    for column_index, sensor_name in enumerate(sensor_columns):
        feature_row[f"{sensor_name}_last"] = to_python_float(last_values[column_index])
        feature_row[f"{sensor_name}_mean"] = to_python_float(mean_values[column_index])
        feature_row[f"{sensor_name}_std"] = to_python_float(std_values[column_index])
        feature_row[f"{sensor_name}_delta"] = to_python_float(delta_values[column_index])

    for spec_name, spec_value in spec_row.items():
        feature_row[str(spec_name)] = spec_value

    return feature_row


def optimize_feature_types(feature_frame: pd.DataFrame) -> pd.DataFrame:
    optimized = feature_frame.copy()
    for column in optimized.select_dtypes(include=["float64"]).columns:
        optimized[column] = optimized[column].astype("float32")
    for column in optimized.select_dtypes(include=["int64"]).columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
    return optimized.sort_values(["vehicle_id", "last_time_step"]).reset_index(drop=True)


def to_python_float(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def compute_snapshot_statistics(sensor_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_counts = np.sum(~np.isnan(sensor_values), axis=0)
    value_sums = np.nansum(sensor_values, axis=0)
    mean_values = np.divide(
        value_sums,
        valid_counts,
        out=np.full(sensor_values.shape[1], np.nan, dtype=float),
        where=valid_counts > 0,
    )
    squared_error = np.square(sensor_values - mean_values)
    variance = np.divide(
        np.nansum(squared_error, axis=0),
        valid_counts,
        out=np.full(sensor_values.shape[1], np.nan, dtype=float),
        where=valid_counts > 0,
    )
    std_values = np.sqrt(variance)
    return mean_values, std_values

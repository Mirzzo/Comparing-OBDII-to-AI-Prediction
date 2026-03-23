from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CLASS_TO_WINDOW = {
    0: ">48 time steps",
    1: "24-48 time steps",
    2: "12-24 time steps",
    3: "6-12 time steps",
    4: "0-6 time steps",
}

CLASS_TO_RISK_BAND = {
    0: "Normal",
    1: "Monitor",
    2: "Advisory",
    3: "Warning",
    4: "Critical",
}

CLASS_TO_REACTIVE_STATUS = {
    0: "Clear",
    1: "Monitor",
    2: "Pending fault",
    3: "Confirmed fault",
    4: "Critical fault",
}

CLASS_TO_RECOMMENDATION = {
    0: "No immediate action",
    1: "Monitor vehicle",
    2: "Schedule inspection",
    3: "Service soon",
    4: "Immediate service",
}


@dataclass(frozen=True)
class ReactiveBaselineExplainer:
    feature_columns: list[str]
    reference_mean: np.ndarray
    reference_std: np.ndarray
    normal_z: float

    @classmethod
    def load(cls, project_root: Path, artifacts_dir: Path) -> "ReactiveBaselineExplainer":
        src_dir = project_root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        import joblib

        model_path = artifacts_dir / "models" / "reactive_baseline.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                "Reactive baseline model not found. Run `python main.py run` first."
            )

        model = joblib.load(model_path)
        return cls(
            feature_columns=list(model.feature_columns_),
            reference_mean=np.asarray(model.reference_mean_, dtype=float),
            reference_std=np.clip(np.asarray(model.reference_std_, dtype=float), 1e-6, None),
            normal_z=float(getattr(model, "normal_z", 3.0)),
        )

    def explain_predictions(
        self,
        feature_frame: pd.DataFrame,
        prediction_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        base_frame = feature_frame.loc[:, ["vehicle_id", *self.feature_columns]].copy()
        merged = prediction_frame.merge(base_frame, on="vehicle_id", how="inner")
        numeric_frame = merged.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        values = numeric_frame.to_numpy(dtype=float, copy=False)
        z_scores = np.abs(values - self.reference_mean) / self.reference_std
        exceedances = np.clip(z_scores - self.normal_z, 0.0, None)

        detail_rows: list[dict[str, object]] = []
        for row_index, prediction_row in enumerate(merged.itertuples(index=False)):
            predicted_label = int(prediction_row.predicted_label)
            true_label = int(prediction_row.true_label)
            row_z_scores = z_scores[row_index]
            row_exceedances = exceedances[row_index]

            pending_count = int(np.sum(row_exceedances > 0.0))
            confirmed_count = int(np.sum(row_exceedances > 1.0))
            severe_count = int(np.sum(row_exceedances > 2.0))
            anomaly_score = float(np.nansum(row_exceedances))
            mil_on = int(confirmed_count >= 2 or severe_count >= 1)
            rule_risk_band, rule_status, rule_recommendation = derive_rule_status(
                pending_count=pending_count,
                confirmed_count=confirmed_count,
                severe_count=severe_count,
            )

            top_signal_text = format_top_signals(self.feature_columns, row_z_scores, row_exceedances)
            top_family_text = format_top_families(self.feature_columns, row_exceedances)

            detail_rows.append(
                {
                    "vehicle_id": int(prediction_row.vehicle_id),
                    "true_label": true_label,
                    "true_window": CLASS_TO_WINDOW[true_label],
                    "true_risk_band": CLASS_TO_RISK_BAND[true_label],
                    "reactive_obdii_prediction": predicted_label,
                    "reactive_bucket_window": CLASS_TO_WINDOW[predicted_label],
                    "reactive_bucket_risk_band": CLASS_TO_RISK_BAND[predicted_label],
                    "reactive_bucket_status": CLASS_TO_REACTIVE_STATUS[predicted_label],
                    "reactive_bucket_recommendation": CLASS_TO_RECOMMENDATION[predicted_label],
                    "rule_risk_band": rule_risk_band,
                    "rule_status": rule_status,
                    "rule_recommendation": rule_recommendation,
                    "reactive_anomaly_score": round(anomaly_score, 6),
                    "mil_status": "On" if mil_on else "Off",
                    "pending_issue_count": pending_count,
                    "confirmed_issue_count": confirmed_count,
                    "severe_issue_count": severe_count,
                    "rule_trigger_summary": (
                        f"pending={pending_count}, confirmed={confirmed_count}, severe={severe_count}"
                    ),
                    "bucket_rule_alignment": "Aligned"
                    if CLASS_TO_RISK_BAND[predicted_label] == rule_risk_band
                    else "Mismatch",
                    "top_triggered_signals": top_signal_text,
                    "top_triggered_families": top_family_text,
                    "reactive_explanation": build_reactive_explanation(
                        predicted_label=predicted_label,
                        mil_on=mil_on,
                        pending_count=pending_count,
                        confirmed_count=confirmed_count,
                        severe_count=severe_count,
                        top_signals=top_signal_text,
                        top_families=top_family_text,
                    ),
                }
            )

        return pd.DataFrame(detail_rows)


def format_top_signals(
    feature_columns: list[str],
    z_scores: np.ndarray,
    exceedances: np.ndarray,
    limit: int = 3,
) -> str:
    triggered_indexes = np.flatnonzero(exceedances > 0.0)
    if len(triggered_indexes) == 0:
        return "No threshold exceedances"

    ordered_indexes = triggered_indexes[np.argsort(exceedances[triggered_indexes])[::-1][:limit]]
    parts = []
    for column_index in ordered_indexes:
        signal_name = feature_columns[column_index].removesuffix("_last")
        parts.append(f"{signal_name} ({z_scores[column_index]:.1f} sigma)")
    return ", ".join(parts)


def format_top_families(
    feature_columns: list[str],
    exceedances: np.ndarray,
    limit: int = 3,
) -> str:
    family_scores: dict[str, float] = {}
    for column_name, exceedance in zip(feature_columns, exceedances, strict=False):
        if not np.isfinite(exceedance) or exceedance <= 0.0:
            continue
        family_name = get_signal_family_name(column_name)
        family_scores[family_name] = family_scores.get(family_name, 0.0) + float(exceedance)

    if not family_scores:
        return "No dominant signal families"

    sorted_families = sorted(
        family_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return ", ".join(f"{family}" for family, _ in sorted_families)


def get_signal_family_name(column_name: str) -> str:
    base_name = column_name.removesuffix("_last")
    family_id = base_name.split("_", maxsplit=1)[0]
    return f"Sensor family {family_id}"


def build_reactive_explanation(
    predicted_label: int,
    mil_on: int,
    pending_count: int,
    confirmed_count: int,
    severe_count: int,
    top_signals: str,
    top_families: str,
) -> str:
    model_band = CLASS_TO_RISK_BAND[predicted_label]
    model_window = CLASS_TO_WINDOW[predicted_label]
    rule_band, rule_status, recommendation = derive_rule_status(
        pending_count=pending_count,
        confirmed_count=confirmed_count,
        severe_count=severe_count,
    )

    if pending_count == 0:
        if predicted_label == 0:
            return (
                "No last-value signal exceeded the learned reactive threshold, so the "
                "rule-level OBD-style status stays clear."
            )
        return (
            f"The saved reactive model bucket is {model_band} ({model_window}), but the "
            "rule-level OBD-style trigger count is clear because no last-value signal "
            "crossed the threshold. "
            "This mismatch reflects weak reactive calibration rather than a literal OBD-II fault trigger."
        )

    explanation = (
        f"Rule-level OBD-style status is '{rule_status}' with MIL {'On' if mil_on else 'Off'}. "
        f"{pending_count} signals exceeded the reactive threshold, {confirmed_count} were strongly abnormal, "
        f"and {severe_count} were severe. Main triggers: {top_signals}. "
        f"Dominant families: {top_families}. Suggested action: {recommendation}."
    )
    if rule_band != model_band:
        explanation += (
            f" The saved reactive model bucket is {model_band} ({model_window}), so the "
            "rule view and model bucket should be interpreted together."
        )
    return explanation


def derive_rule_status(
    pending_count: int,
    confirmed_count: int,
    severe_count: int,
) -> tuple[str, str, str]:
    if pending_count == 0:
        return "Normal", "Clear", "No immediate action"
    if severe_count >= 1 or confirmed_count >= 2:
        return "Critical", "Confirmed fault", "Immediate service"
    if confirmed_count >= 1:
        return "Warning", "Pending fault", "Service soon"
    return "Monitor", "Monitor", "Monitor vehicle"

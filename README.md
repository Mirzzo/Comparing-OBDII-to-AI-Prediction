# AI Maintenance Prediction System

This repository contains a Python-based predictive maintenance workflow for the SCANIA Component X dataset stored under `Dataset/data`.

The system does four things:

1. Streams the raw operational readouts vehicle by vehicle.
2. Builds snapshot-level training examples from the training split.
3. Trains three classifiers:
   - a reactive threshold baseline
   - Logistic Regression with minority-class oversampling
   - Random Forest with class-balanced bootstrapping
4. Evaluates them on the provided validation and test splits and saves reports to `artifacts/`.

The evaluation report includes both standard classification metrics and the challenge-style asymmetric maintenance cost.

## Dataset framing

The dataset documentation defines five classes for the last observed readout of a vehicle:

- `0`: more than 48 time steps before failure
- `1`: 48 to 24 time steps before failure
- `2`: 24 to 12 time steps before failure
- `3`: 12 to 6 time steps before failure
- `4`: 6 to 0 time steps before failure

The training split does not ship with these five labels directly. Instead, this project infers them from `train_tte.csv` by comparing each selected readout time with `length_of_study_time_step`. For vehicles with `in_study_repair = 0`, the final 48 time steps are excluded from training because they are right-censored and cannot be assigned a reliable failure window.

To handle the strong class imbalance, the training pipeline now oversamples minority classes for Logistic Regression while keeping Random Forest on class-balanced bootstrapping. This improves the minority-class sensitivity of the linear model without forcing the same strategy on every estimator.

## Project layout

- `main.py`: simple entry point
- `src/maintenance_prediction/features.py`: dataset streaming and feature creation
- `src/maintenance_prediction/baseline.py`: reactive rule-based baseline
- `src/maintenance_prediction/modeling.py`: training, evaluation, and report export
- `src/maintenance_prediction/cli.py`: command-line interface
- `obdii_comparison/`: separate OBD-II-style comparison workbook generator

## Setup

Install Python 3.10+ and then install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Build cached feature tables:

```bash
python main.py prepare-features
```

Run the full experiment:

```bash
python main.py run
```

Reuse cached feature tables on later runs:

```bash
python main.py run --reuse-features
```

Generate the standalone OBD-II comparison workbook without changing the AI
training pipeline:

```bash
python -m pip install -r obdii_comparison/requirements.txt
python -m obdii_comparison.main --ai-model logistic_regression
```

Optional arguments:

- `--dataset-root Dataset`
- `--output-dir artifacts`
- `--chunksize 50000`

## Outputs

After `run`, the project writes:

- `artifacts/features/*.csv`: engineered train/validation/test features
- `artifacts/models/*.joblib`: fitted models
- `artifacts/reports/metrics.json`: summary metrics
- `artifacts/reports/*_confusion_matrix_*.csv`: confusion matrices
- `artifacts/reports/*_predictions_*.csv`: per-vehicle predictions

## Notes

- The operational files are large, so the feature builder reads them in chunks instead of loading the full training CSV into memory.
- The code assumes the operational readout files are ordered by `vehicle_id` and `time_step`, which matches the supplied dataset structure.
- The raw dataset and generated artifacts are intentionally ignored in Git so the repository stays lightweight and the comparison tooling remains reproducible from local files.

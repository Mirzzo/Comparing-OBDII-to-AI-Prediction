# AI Maintenance Prediction System

This repository contains a Python-based predictive maintenance workflow for the SCANIA Component X dataset stored under `Dataset/data`.

The system does five things:

1. Streams the raw operational readouts vehicle by vehicle.
2. Builds snapshot-level training examples from the training split.
3. Trains the main benchmark models:
   - a reactive threshold baseline
   - Logistic Regression with expected-cost decoding
   - Random Forest with expected-cost decoding
   - single-stage CatBoost
   - two-stage CatBoost
4. Evaluates them on the provided validation and test splits and saves reports to `artifacts/`.
5. Supports separate comparison and experiment workflows without overwriting the main benchmark outputs.

The evaluation report includes both standard classification metrics and the challenge-style asymmetric maintenance cost.

## Dataset framing

The dataset documentation defines five classes for the last observed readout of a vehicle:

- `0`: more than 48 time steps before failure
- `1`: 48 to 24 time steps before failure
- `2`: 24 to 12 time steps before failure
- `3`: 12 to 6 time steps before failure
- `4`: 6 to 0 time steps before failure

The training split does not ship with these five labels directly. Instead, this project infers them from `train_tte.csv` by comparing each selected readout time with `length_of_study_time_step`. For vehicles with `in_study_repair = 0`, the final 48 time steps are excluded from training because they are right-censored and cannot be assigned a reliable failure window.

The main benchmark keeps the original dataset split intact and uses cost-aware prediction decoding for Logistic Regression and Random Forest. The CatBoost variants use `auto_class_weights="SqrtBalanced"` to handle imbalance. Separate class-weighted, oversampling, and binary-risk experiments are available through `python main.py run-experiments` and are written to `artifacts/experiments/` without replacing the main benchmark outputs.

## Project layout

- `main.py`: simple entry point
- `src/maintenance_prediction/features.py`: dataset streaming and feature creation
- `src/maintenance_prediction/baseline.py`: reactive rule-based baseline
- `src/maintenance_prediction/modeling.py`: training, evaluation, and report export
- `src/maintenance_prediction/cli.py`: command-line interface
- `src/maintenance_prediction/experiments.py`: separate class-weighted, oversampling, and binary-risk experiment runner
- `obdii_comparison/`: separate OBD-II-style comparison workbook generator
- `scripts/`: reproducible figure generators used for the paper

## Dataset download

This repository does not include the raw SCANIA Component X dataset.

Download it before running the project:

https://researchdata.se/en/catalogue/dataset/2024-34

The dataset is not committed to Git because it is large and externally hosted.
You need to download it manually and place the extracted files under `Dataset/`
so the project can find `Dataset/data/*.csv` and `Dataset/documentation/*`.

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

Run the separate training-improvement experiments without overwriting the main
benchmark outputs:

```bash
python main.py run-experiments --reuse-features
```

This writes class-weighted multiclass, moderate-oversampling multiclass, and
binary-risk experiment outputs under `artifacts/experiments/`, including
comparison CSV files against the previously saved benchmark.

Generate the standalone OBD-II comparison workbook without changing the AI
training pipeline:

```bash
python -m pip install -r obdii_comparison/requirements.txt
python -m obdii_comparison.main
```

By default, the comparison tool selects the AI model with the lowest test mean
challenge cost from `artifacts/reports/metrics.json`. To force the current main
comparison model explicitly:

```bash
python -m obdii_comparison.main --ai-model catboost_two_stage
```

Generate the paper figures used for the binary class distribution and the
main-comparison line charts:

```bash
python scripts/generate_binary_class_chart.py
python scripts/generate_main_comparison_table_charts.py
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
- `artifacts/experiments/*`: separate experiment results from `run-experiments`
- `obdii_comparison/artifacts/Comparison Table.xlsx`: standalone reactive-vs-AI workbook

## Notes

- The operational files are large, so the feature builder reads them in chunks instead of loading the full training CSV into memory.
- The code assumes the operational readout files are ordered by `vehicle_id` and `time_step`, which matches the supplied dataset structure.
- The raw dataset and generated artifacts are intentionally ignored in Git so the repository stays lightweight and the comparison tooling remains reproducible from local files.
- The dataset must be downloaded manually before `python main.py prepare-features` or `python main.py run` will work.

# OBD-II Comparison Tool

This folder contains a standalone comparison utility that is intentionally kept
separate from the AI prediction system in `src/maintenance_prediction`.

It does not retrain, modify, or replace the AI models. Instead, it:

1. Reads the existing feature tables and prediction outputs from `artifacts/`.
2. Reuses the saved `reactive_baseline` results as the canonical
   `Reactive OBD-II-style baseline`.
3. Adds an OBD-style explanation layer on top of the saved reactive baseline so
   the baseline output is easier to read.
4. Compares that reactive baseline against the selected AI model.
5. Writes `Comparison Table.xlsx` plus standalone baseline comparison outputs to
   `obdii_comparison/artifacts/`.

If `--ai-model` is not provided, the tool selects the AI model with the lowest
test mean challenge cost from `artifacts/reports/metrics.json`. With the current
saved benchmark artifacts, that default main-comparison model is usually
`catboost_two_stage`.

## Important note

This tool does **not** add real OBD-II logs to the SCANIA dataset. It produces
a reporting-only comparison that treats the saved reactive baseline as the
paper's `Reactive OBD-II-style baseline`. In the paper, this should still be
described as a proxy representation of reactive OBD-II logic, not as real
scanner output.

## What the workbook adds

The comparison workbook keeps the original saved baseline metrics intact, but it
also adds a more understandable OBD-style detail view:

- `reactive_bucket_*`: the original saved reactive baseline class output
- `rule_*`: a threshold-count interpretation of the same row in more OBD-style language
- `mil_status`: a proxy lamp indicator based on rule severity
- `pending_issue_count`, `confirmed_issue_count`, `severe_issue_count`: count-style trigger fields
- `top_triggered_signals` and `top_triggered_families`: the strongest abnormal signals
- `reactive_explanation`: a plain-language explanation for that row

If `bucket_rule_alignment` is `Mismatch`, the saved reactive class bucket and
the rule-style trigger view are not telling the exact same story for that row.
That is expected in this proxy setup and should be interpreted as a reactive
calibration issue, not as literal ECU behavior.

## Usage

Install the extra Excel dependency if needed:

```bash
python -m pip install -r obdii_comparison/requirements.txt
```

Run the comparison tool from the repository root:

```bash
python -m obdii_comparison.main
```

Explicitly select the current main-comparison AI model:

```bash
python -m obdii_comparison.main --ai-model catboost_two_stage
```

Alternative comparisons are still supported:

```bash
python -m obdii_comparison.main --ai-model logistic_regression
```

The workbook is written to:

`obdii_comparison/artifacts/Comparison Table.xlsx`

If the workbook is already open in Excel, the tool writes:

`obdii_comparison/artifacts/Comparison Table (Refreshed).xlsx`

## Workbook contents

The workbook currently includes these sheets:

- `Summary`
- `Baseline vs AI`
- `Class Distribution`
- `Decision Legend`
- `Interpretation`
- `Validation Predictions`
- `Test Predictions`
- `Validation Baseline CM`
- `Test Baseline CM`

## Additional outputs

The tool also writes standalone baseline exports such as:

- `reactive_obdii_baseline_metrics.json`
- `reactive_obdii_baseline_validation_predictions.csv`
- `reactive_obdii_baseline_test_predictions.csv`
- `reactive_obdii_baseline_validation_details.csv`
- `reactive_obdii_baseline_test_details.csv`

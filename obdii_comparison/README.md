# OBD-II Comparison Tool

This folder contains a standalone comparison utility that is intentionally kept
separate from the AI prediction system in `src/maintenance_prediction`.

It does not retrain, modify, or replace the AI models. Instead, it:

1. Reads the existing feature tables and prediction outputs from `artifacts/`.
2. Reuses the saved `reactive_baseline` results as the canonical
   `Reactive OBD-II-style baseline`.
3. Compares that reactive baseline against the selected AI model.
4. Writes `Comparison Table.xlsx` plus standalone baseline comparison outputs to
   `obdii_comparison/artifacts/`.

## Important note

This tool does **not** add real OBD-II logs to the SCANIA dataset. It produces
a reporting-only comparison that treats the saved reactive baseline as the
paper's `Reactive OBD-II-style baseline`. In the paper, this should still be
described as a proxy representation of reactive OBD-II logic, not as real
scanner output.

## Usage

Install the extra Excel dependency if needed:

```bash
python -m pip install -r obdii_comparison/requirements.txt
```

Run the comparison tool from the repository root:

```bash
python -m obdii_comparison.main
```

Optional:

```bash
python -m obdii_comparison.main --ai-model logistic_regression
```

The workbook is written to:

`obdii_comparison/artifacts/Comparison Table.xlsx`

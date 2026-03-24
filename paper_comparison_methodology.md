# AI vs Reactive OBD-II-Style Comparison Methodology

## Purpose of this document

This note explains how the comparison in this repository is calculated and how it
should be described in the paper. It is based on the current codebase, not on
external notes or draft text files.

The project compares:

- an AI-based predictive maintenance system trained on the SCANIA Component X dataset
- a separate reactive rule-based baseline used as a proxy for conventional OBD-II-style diagnostics

This distinction matters. The dataset is not a native passenger-car OBD-II log
dataset, so the reactive side should be described as an **OBD-II-style proxy**
or **reactive OBD-II-style baseline**, not as real scanner output from a vehicle
ECU.

## The SCANIA Component X dataset

The SCANIA Component X dataset used in this repository is a predictive
maintenance dataset centered on an anonymized target component called
**Component X**. It is organized around repeated operational readouts from the
same vehicle over time, combined with static vehicle specifications and
time-to-event or class-label metadata.

In the local project layout, the dataset is expected under `Dataset/data/` and
is split into three parts:

| Split | Main files used in this project | Purpose |
| --- | --- | --- |
| Train | `train_operational_readouts.csv`, `train_specifications.csv`, `train_tte.csv` | Build training snapshots and infer labels from time-to-event information |
| Validation | `validation_operational_readouts.csv`, `validation_specifications.csv`, `validation_labels.csv` | Evaluate model behavior on a held-out split |
| Test | `test_operational_readouts.csv`, `test_specifications.csv`, `test_labels.csv` | Final held-out evaluation |

The dataset combines three kinds of information:

When the outputs in this project contain a `split` column, the meaning is:

- `validation`: the held-out development set used to compare model behavior and decide which approach is more promising before the final comparison
- `test`: the final held-out set used for the main unbiased performance check after the modeling setup has been chosen

### 1. Longitudinal operational readouts

The operational files contain repeated sensor measurements over time. Each row
includes at least:

- `vehicle_id`
- `time_step`
- many anonymized sensor columns such as `171_0`, `666_0`, `427_0`, `397_34`, and similar fields

This means the dataset is not a single flat table with one row per vehicle. It
is a time-series dataset in which each vehicle has a sequence of readouts across
multiple time steps.

The operational files are also large, which is why the feature builder reads
them in chunks instead of loading everything into memory at once. In the local
copy used here, `train_operational_readouts.csv` is roughly 1.2 GB, and the
validation and test operational files are each about 215 MB.

### 2. Static vehicle specifications

The specification files contain one row per vehicle and store categorical
vehicle-level descriptors such as `Spec_0` through `Spec_7`. These fields do not
change over time and provide context that may influence failure behavior or
operating conditions.

In this repository, those `Spec_*` fields are treated as categorical inputs and
are encoded during model training.

### 3. Event and label metadata

The metadata differs slightly by split:

- `train_tte.csv` provides `length_of_study_time_step` and `in_study_repair`
- `validation_labels.csv` and `test_labels.csv` provide the final `class_label`

The training split therefore does not directly provide the same five-class label
table as the validation and test splits. Instead, the training labels are
constructed from the time-to-event information.

## Why this dataset is suitable for predictive maintenance

The key strength of the dataset is that it combines:

- historical sensor trajectories
- static vehicle context
- outcome information related to failure timing

That makes it suitable for **predictive maintenance**, because the task is not
only to identify abnormal current behavior, but to estimate how close a vehicle
is to an upcoming failure window.

The dataset is also well suited to comparing reactive and predictive approaches:

- the time-series readouts allow the AI system to learn patterns associated with future failure proximity
- the last-value readouts can be used to construct a simpler reactive threshold-based baseline

## Important dataset limitations for the paper

Several limitations should be stated clearly:

- The sensor names are anonymized identifiers, not human-readable engineering variable names.
- The dataset does not provide real OBD-II diagnostic trouble codes, real OBD-II parameter IDs, or actual ECU MIL states.
- The training labels are partially inferred from `train_tte.csv`, rather than being shipped directly as the same five-class classification table used for validation and test.
- The dataset is strongly imbalanced, with class 0 dominating the holdout splits.

Because of these points, the paper should describe the reactive side as an
**OBD-II-style proxy baseline** rather than as a direct evaluation of a real
commercial OBD-II diagnostic system.

## What the AI model predicts

The AI system performs a five-class classification task. Each predicted class
represents a time-to-failure window for Component X:

| Class | Meaning |
| --- | --- |
| 0 | More than 48 time steps before failure |
| 1 | 24 to 48 time steps before failure |
| 2 | 12 to 24 time steps before failure |
| 3 | 6 to 12 time steps before failure |
| 4 | 0 to 6 time steps before failure |

In practical terms, the AI model predicts **how close a vehicle is to a future
failure window**, not just whether a fault is present at the current moment.

For validation and test data, the model makes one prediction per vehicle using
the full history available up to the last recorded time step for that vehicle.

## How the training data is built

The raw operational readouts are converted into one row per vehicle snapshot.
For each selected snapshot, the pipeline creates:

- metadata features such as `history_length`, `first_time_step`, `last_time_step`, `time_step_span`, and `mean_step_interval`
- per-sensor summary features:
  - `sensor_last`
  - `sensor_mean`
  - `sensor_std`
  - `sensor_delta`
- specification features from the `Spec_*` columns

For the training split, snapshots are chosen from the historical trajectory so
that the model sees examples from different time-to-failure windows. The target
snapshot for each class is chosen near these reference deltas:

| Class | Target delta used during snapshot selection |
| --- | --- |
| 0 | 72 time steps before failure |
| 1 | 36 time steps before failure |
| 2 | 18 time steps before failure |
| 3 | 9 time steps before failure |
| 4 | 3 time steps before failure |

If a vehicle does not have an in-study repair event, it only contributes class 0
examples.

## Models used in the prediction system

The main training pipeline currently produces five systems:

1. `reactive_baseline`
2. `logistic_regression`
3. `random_forest`
4. `catboost`
5. `catboost_two_stage`

### Logistic regression

The logistic regression model uses:

- median imputation for numeric features
- standardization for numeric features
- most-frequent imputation plus one-hot encoding for categorical `Spec_*` features

In the current code, logistic regression trains on the natural class
distribution without oversampling. At prediction time, it does not use plain
maximum-probability decoding. Instead, it uses the challenge cost matrix to
choose the class with the lowest expected maintenance cost.

### Random forest

The random forest model uses:

- median imputation for numeric features
- one-hot encoded categorical specification features
- `class_weight="balanced_subsample"`

Like logistic regression, the random forest now trains on the natural class
distribution and uses expected-cost decoding at prediction time.

### CatBoost

The single-stage CatBoost model uses:

- native handling of categorical `Spec_*` features rather than one-hot encoding
- `auto_class_weights="SqrtBalanced"`
- validation-guided early stopping

In the current results, this model reaches the highest raw test accuracy among
the AI systems, but it still behaves mostly like a majority-class classifier on
the minority fault windows.

### Two-stage CatBoost

The two-stage CatBoost model is a hierarchical predictive-maintenance model:

- stage 1 predicts whether the vehicle is class `0` or a future-failure class
  (`1` to `4`)
- stage 2 predicts the specific failure-window class among `1` to `4`
- the final fault threshold is tuned on the validation set to maximize macro F1,
  then lower challenge cost, then higher accuracy

This model is currently the strongest main AI comparison in the repository
because it achieves the best balance between minority detection and practical
error cost. In the current test results, it reaches:

- accuracy = `0.891378`
- macro precision = `0.213186`
- macro recall = `0.250957`
- macro F1 = `0.215768`
- mean challenge cost = `8.954410`

## Why accuracy is not enough

The dataset is highly imbalanced, with class 0 making up the overwhelming
majority of vehicles in validation and test. Because of that, a model can obtain
very high accuracy simply by predicting the majority class most of the time.

For this reason, the project does not rely on accuracy alone. It also reports:

- macro precision
- macro recall
- macro F1
- weighted F1
- confusion matrices
- a challenge cost metric

## Challenge cost used for comparison

The project uses an asymmetric cost matrix. Predicting a vehicle as too healthy
when it is actually near failure is penalized much more heavily than producing
an earlier warning.

This reflects predictive-maintenance logic: missing an imminent failure is
usually worse than scheduling an unnecessary inspection.

The implemented cost matrix is:

| True class \\ Predicted class | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 7 | 8 | 9 | 10 |
| 1 | 200 | 0 | 7 | 8 | 9 |
| 2 | 300 | 200 | 0 | 7 | 8 |
| 3 | 400 | 300 | 200 | 0 | 7 |
| 4 | 500 | 400 | 300 | 200 | 0 |

The training reports both total challenge cost and mean challenge cost.

## What the reactive OBD-II-style engine is

The repository does not contain real OBD-II trouble codes or ECU lamp states.
Instead, it implements a **reactive OBD-II-style baseline** that behaves like a
simple threshold-based diagnostic engine.

The idea is to mimic the logic of conventional reactive diagnostics:

- observe the current sensor state
- compare it with a learned normal reference
- trigger warnings when current values drift beyond a threshold
- map stronger anomalies to more severe states

This is why the baseline is a good conceptual comparison for reactive diagnostic
logic, even though it is not a literal OBD-II scanner.

## What real OBD-II does conceptually

In a real vehicle, OBD-II is an onboard diagnostic framework that monitors
engine and emissions-related behavior through the electronic control unit
(ECU). When monitored conditions exceed defined thresholds or fail a diagnostic
test repeatedly, the system can:

- store a diagnostic trouble code (DTC)
- mark a fault as pending or confirmed
- turn on the malfunction indicator lamp (MIL)
- support inspection and service decisions

That logic is mainly reactive. It is centered on detecting abnormal conditions
that have already crossed a threshold, rather than estimating how far the system
is from a future failure window. The proxy baseline in this repository follows
that same reactive idea, but it does so using the available SCANIA sensor
features instead of true DTCs and PIDs.

## How the reactive baseline is calculated

The reactive baseline uses only the `*_last` features, which means the most
recent sensor reading from each vehicle snapshot.

### Step 1: Learn a healthy reference

During training, the baseline takes the class 0 examples as the healthy
reference set. From those rows, it computes:

- the mean of each `*_last` feature
- the standard deviation of each `*_last` feature

### Step 2: Calculate per-signal deviation

For each vehicle row, it computes a z-score-style deviation:

```text
z = abs(current_value - healthy_mean) / healthy_std
```

Only deviation beyond the normal threshold counts toward the anomaly score:

```text
exceedance = max(z - 3.0, 0)
```

### Step 3: Create one anomaly score

The anomaly score for a row is the sum of exceedances across all `*_last`
features:

```text
anomaly_score = sum(exceedance over all last-value features)
```

This means:

- normal signals contribute `0`
- mildly abnormal signals contribute a small amount
- strongly abnormal signals contribute more

### Step 4: Map anomaly scores to classes

The model calculates the median anomaly score for each class in the training
data and uses the midpoints between those medians as class boundaries.

At prediction time, the anomaly score is placed into one of the five classes
using those learned boundaries.

This produces the saved `reactive_baseline` class prediction, which is kept in
the comparison outputs as the **reactive bucket**.

## How the readable OBD-style view is created

The `obdii_comparison` tool does not retrain any model. It reads the saved
reactive baseline predictions and then adds a more understandable OBD-style
interpretation layer on top.

That readable layer includes:

- `reactive_bucket_*`: the original saved reactive prediction class
- `rule_*`: a rule-style interpretation of the same vehicle based on signal trigger counts
- `mil_status`: a proxy lamp indicator
- `pending_issue_count`
- `confirmed_issue_count`
- `severe_issue_count`
- `top_triggered_signals`
- `top_triggered_families`
- `reactive_explanation`

### Rule-style interpretation

The readable rule layer uses the same exceedance values and counts:

- `pending_issue_count`: number of signals with exceedance greater than `0`
- `confirmed_issue_count`: number of signals with exceedance greater than `1`
- `severe_issue_count`: number of signals with exceedance greater than `2`

The proxy MIL state is turned on when:

- `confirmed_issue_count >= 2`, or
- `severe_issue_count >= 1`

The rule-style status is then assigned as:

| Condition | Rule risk band | Rule status | Recommendation |
| --- | --- | --- | --- |
| No pending triggers | Normal | Clear | No immediate action |
| At least one severe trigger or at least two confirmed triggers | Critical | Confirmed fault | Immediate service |
| At least one confirmed trigger | Warning | Pending fault | Service soon |
| Otherwise | Monitor | Monitor | Monitor vehicle |

This is designed to make the reactive baseline easier to interpret in a paper,
because it looks more like a diagnostic decision process.

## Why there can be a mismatch between bucket and rule view

The saved reactive baseline class is learned from anomaly-score boundaries. The
readable rule view is based on trigger counts. Those two views are related, but
they are not identical.

As a result, some rows may show:

- a severe saved reactive bucket, but
- a mild or clear rule-style interpretation

In the comparison workbook, this is reported through the
`bucket_rule_alignment` column.

This mismatch should be described as a **reactive calibration limitation**, not
as a contradiction in the AI model. The saved bucket is kept because it preserves
metric consistency with the original benchmark results. The rule view is added
only to make the baseline more understandable.

## How the standalone comparison workbook is calculated

The comparison workbook is generated by `obdii_comparison/main.py`. It is
completely separate from the training pipeline and does not modify the trained
models.

The workflow is:

1. Load `validation_features.csv` and `test_features.csv` from `artifacts/features/`
2. Load `metrics.json` and the saved prediction CSV files from `artifacts/reports/`
3. Load the saved `reactive_baseline.joblib` model only to compute readable baseline details
4. Select the AI model for comparison
5. Merge the baseline details with the AI predictions by `vehicle_id`
6. Write the comparison workbook and standalone CSV outputs to `obdii_comparison/artifacts/`

### How the AI model is selected for comparison

The comparison tool can compare the reactive baseline against:

- logistic regression
- random forest
- CatBoost
- Two-stage CatBoost

If no model is specified manually, it chooses the AI model with the **lowest
test mean challenge cost** from the saved metrics.

With the current results, that default model is **Two-stage CatBoost**.

## What the AI model means in the workbook

In the comparison workbook, the AI prediction is the model's estimated
time-to-failure class for that vehicle. The workbook adds:

- `ai_prediction`
- `ai_window`
- `ai_risk_band`
- `ai_status`
- `ai_recommendation`

These are presentation fields added to make the AI decision easier to compare
with the reactive OBD-II-style baseline.

## Why Two-stage CatBoost is the main AI comparison

The current results show five different patterns:

- the reactive baseline performs very poorly
- cost-sensitive logistic regression and random forest become very aggressive
  toward the near-failure class and lose too much overall accuracy
- single-stage CatBoost has the highest test accuracy (`0.966700`), but it still
  misses most minority failure windows
- Two-stage CatBoost preserves high overall accuracy while improving minority
  sensitivity and achieving the lowest mean challenge cost (`8.954410`)

This is why Two-stage CatBoost is the preferred AI model for the main
OBD-II-style comparison in the paper.

The argument is not that Two-stage CatBoost is universally the best classifier.
The argument is that, under this dataset and this cost structure, it provides
the most useful predictive-maintenance tradeoff among the models currently
implemented in the repository.

## How to describe the comparison in the paper

The cleanest phrasing is:

> The study compares a reactive rule-based baseline representing conventional
> OBD-II-style diagnostic logic against an AI-based predictive maintenance model.

Important wording choices:

- do not claim that the project uses real OBD-II logs
- do not claim that the reactive baseline is a commercial scanner
- describe the baseline as a reactive threshold-based OBD-II-style proxy
- describe the AI system as a predictive model that estimates time-to-failure class

## Recommended paper interpretation

The comparison should be interpreted as follows:

- The reactive baseline represents conventional reactive diagnostics, which are simple and interpretable but weak for early predictive detection.
- The AI system uses the historical sensor trajectory and specification context to estimate proximity to failure.
- Single-stage CatBoost can appear very strong on accuracy because the dataset is highly imbalanced and class 0 dominates the holdout splits.
- Two-stage CatBoost is the more meaningful main comparison because it improves minority-class sensitivity and produces the lowest challenge cost among the available AI models while keeping high overall accuracy.
- The readable OBD-style layer improves interpretability of the baseline, but it remains a proxy rather than real ECU-level OBD-II output.

## Files related to the comparison

Main implementation files:

- `src/maintenance_prediction/features.py`
- `src/maintenance_prediction/modeling.py`
- `src/maintenance_prediction/baseline.py`
- `obdii_comparison/main.py`
- `obdii_comparison/reactive_details.py`

Main outputs:

- `artifacts/reports/metrics.json`
- `artifacts/reports/catboost_two_stage_test_predictions.csv`
- `obdii_comparison/artifacts/Comparison Table.xlsx`
- `obdii_comparison/artifacts/Comparison Table (Refreshed).xlsx`
- `obdii_comparison/artifacts/reactive_obdii_baseline_test_details.csv`

## Short version for the paper

If a short explanation is needed in the methodology section, the comparison can
be summarized like this:

The AI system predicts one of five time-to-failure classes for each vehicle,
ranging from more than 48 time steps before failure to 0-6 time steps before
failure. A separate reactive threshold-based baseline is used to represent
conventional OBD-II-style diagnostic logic. That baseline learns normal
last-value sensor behavior from healthy class 0 examples, measures current
deviation from that reference, and maps anomaly scores into the same five
classes. A standalone comparison tool then combines the saved baseline outputs
with the AI predictions and presents them in an OBD-style readable format with
proxy MIL state, trigger counts, risk bands, and row-level explanations. In the
current results, Two-stage CatBoost is the preferred AI comparison model
because it achieves the lowest mean challenge cost and the strongest macro-F1
among the available AI models while still retaining high overall accuracy.

# File Formats

This page describes every file involved in the pipeline — inputs, configuration files, generalization hierarchies, and all generated outputs.

---

## Source data

### `data/adult.csv`

The original UCI Adult dataset. This file is **never modified** — it serves as the reference throughout the entire pipeline.

| Field | Description |
|---|---|
| Format | CSV, comma-separated |
| Rows | ~30,000 records |
| Columns | 14 attributes (age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income) |
| Role | Ground truth for all utility comparisons |

---

### `data/adult_with_record_id.csv`

A copy of `adult.csv` with an added `record_id` column (unique integer per row). Used by linkage and MIA attack scripts that need to track individual records across transformations.

---

## Configuration files

### `configs/base_config.json`

Base configuration shared across all experiments. Defines the dataset path, quasi-identifiers, sensitive attribute, and default privacy parameters.

```json
{
  "dataset": "data/adult.csv",
  "quasi_identifiers": ["age", "sex", "race", "marital-status", "native-country"],
  "sensitive_attributes": ["income"],
  "k": 5,
  "l": 2,
  "suppression_limit": 10,
  "utility_measure": "loss",
  "utility_aggregate": "arithmetic_mean",
  "backend": "arx"
}
```

---

### `configs/benchmark_grid.json`

Defines the parameter grid used by `run_benchmark.py` to generate all experiment combinations.

```json
{
  "qi_subset_sizes": [2, 3, 4, 5],
  "k_values": [2, 5, 10, 20],
  "l_values": [2],
  "suppression_limits": [10]
}
```

Each combination of parameters generates one independent experiment.

---

### `configs/classification_config.json`

Configuration for the classification utility benchmark. Specifies the number of folds, the target attribute, the active models, and their hyperparameters.

```json
{
  "n_folds": 10,
  "random_state": 42,
  "target": "income",
  "active_models": ["naive_bayes"],
  "feature_cols": ["age", "sex", "race", "marital-status", "native-country"]
}
```

All parameters — including hyperparameters for each model — are fully configurable in this file.

---

### `outputs/configs/{experiment_id}.json`

Runtime configuration generated for each experiment. It is a resolved copy of `base_config.json` with all paths made absolute, the hierarchy mapping filled in, and the exact parameters used for that run. Serves as a reproducibility record.

---

## Generalization hierarchies

### `hierarchies/{attribute}.csv`

One CSV file per quasi-identifier. Each row defines one value and its successive generalizations, from the most specific (level 0) to the most general (usually `*`).

**Example — `hierarchies/age.csv`:**

```
25;25-29;20-29;*
26;25-29;20-29;*
27;25-29;20-29;*
...
```

| Column index | Meaning |
|---|---|
| 0 | Original value |
| 1 | Level 1 generalization |
| 2 | Level 2 generalization |
| … | … |
| Last | Full suppression (`*`) |

Available hierarchy files: `age.csv`, `sex.csv`, `race.csv`, `marital-status.csv`, `native-country.csv`, `education.csv`, `occupation.csv`, `workclass.csv`, `relationship.csv`.

---

## Anonymization outputs

### `outputs/anonymized/{experiment_id}.csv`

The public anonymized dataset produced by ARX for a given experiment. Quasi-identifier values are replaced by generalized values or `*` (suppressed). The sensitive attribute (`income`) is kept unchanged.

This is the file used as input for all utility evaluations.

---

### `outputs/anonymized_eval/{experiment_id}.csv`

Internal version of the anonymized dataset that retains additional columns used during evaluation (e.g. fold indices, record identifiers). Not published — used only by evaluation scripts.

---

### `outputs/metrics/{experiment_id}.json`

ARX result metrics collected after each anonymization. Contains:

- Equivalence class statistics (count, min/avg/max size, suppressed records)
- Per-attribute metrics (granularity, non-uniform entropy, generalization intensity, squared error)
- Global metrics (discernibility, ambiguity, SSESST, record-level squared error)
- Optimization scores (min and max)
- Anonymization time and number of transformations explored

---

### `outputs/benchmark_summary.csv`

Master summary table — one row per experiment. Aggregates key parameters and metrics from all completed experiments. Used as the entry point by classification and reporting scripts.

| Column | Description |
|---|---|
| `experiment_id` | Unique experiment identifier |
| `status` | `success` or `error` |
| `quasi_identifiers` | Pipe-separated list of QI used |
| `k`, `l`, `t` | Privacy model parameters |
| `suppression_limit` | Max suppression percentage |
| `csv_path` | Path to the anonymized CSV |
| `config_path` | Path to the runtime config |
| + ARX metrics | All metrics from `outputs/metrics/` |

---

## Classification outputs

### `outputs/classification/{experiment_id}_classification.json`

Full classification results for one experiment. Contains, for each model (ZeroR, LR, NB, RF):

- Global accuracy (input and output)
- Relative accuracy
- Per-class sensitivity and specificity
- AUC and ROC curve points (100 sampled)
- Brier score and Brier skill score

---

### `outputs/classification_summary.csv`

Flattened summary of classification results across all experiments — one row per experiment per model. Used for cross-experiment comparison.

---

## File classification

| File type | Perspective | Usage |
|---|---|---|
| `data/adult.csv` | Reference | Never modified — ground truth |
| `configs/*.json` | Configuration | Define experiments and models |
| `hierarchies/*.csv` | Input | Define generalization trees |
| `outputs/anonymized/*.csv` | Public output | Result of anonymization |
| `outputs/metrics/*.json` | Internal | ARX analysis metrics |
| `outputs/benchmark_summary.csv` | Internal | Experiment index |
| `outputs/classification/*.json` | Internal | Classification evaluation results |
| `outputs/configs/*.json` | Internal | Reproducibility record |

# Quality Models

Quality models are objective functions that ARX minimizes when searching for the optimal transformation. They quantify the information loss caused by generalization and suppression.

---

## Role

During anonymization, ARX explores the space of possible transformations and selects the one that satisfies the privacy constraints **while minimizing the loss measured by the chosen quality model**.

The choice of quality model directly influences which transformation ARX considers "the best".

---

## Available quality models

### `loss` — Information Loss *(default)*

Measures the overall information loss caused by generalization. For each attribute, it computes the ratio between the size of the generalized group and the total size of the hierarchy.

- The closer to **0**, the less information is lost.
- Compatible with [aggregate functions](fonctions_agregats.md).

---

### `height` — Hierarchy Height

Measures the average generalization level applied within the hierarchies. A high level means that values have been strongly generalized (pushed high up in the generalization tree).

- Favors transformations that stay close to the original values.
- Compatible with [aggregate functions](fonctions_agregats.md).

---

### `precision` — Precision

Measures the granularity of values retained in the anonymized dataset. High precision indicates that values have only been slightly generalized.

$$\text{precision}(a) = 1 - \frac{\text{generalization level}}{\text{maximum hierarchy height}}$$

- Compatible with [aggregate functions](fonctions_agregats.md).

---

### `nm_entropy` — Non-uniform Entropy

Measures the entropy of attribute distributions while accounting for the actual frequencies of values. Unlike classical entropy, it weights each value by its frequency in the original data.

- More sensitive to distribution imbalances than classical entropy.
- Compatible with [aggregate functions](fonctions_agregats.md).

---

### `entropy` — Entropy

Measures the information-theoretic entropy of the anonymized dataset. It quantifies the uncertainty introduced by generalization.

- Not compatible with aggregate functions.

---

### `discernibility` — Discernibility

Penalizes large equivalence classes. It computes the sum of the squares of all equivalence class sizes.

$$\text{discernibility} = \sum_{G \in \mathcal{G}} |G|^2$$

- A lower value indicates smaller groups and therefore better data precision.
- Not compatible with aggregate functions.

---

### `aecs` — Average Equivalence Class Size

Measures the average size of equivalence classes in the anonymized dataset.

$$\text{AECS} = \frac{n}{\text{number of equivalence classes}}$$

- A simple indicator of anonymization granularity.
- Not compatible with aggregate functions.

---

### `ambiguity` — Ambiguity

Measures the degree of ambiguity of each record — i.e., the number of possible values for each attribute after generalization.

- Not compatible with aggregate functions.

---

### `classification` — Classification Metric

Utility metric oriented towards preserving utility for supervised classification tasks.

- Not compatible with aggregate functions.

---

## Summary

| Model | Short description | Aggregate supported |
|---|---|:---:|
| `loss` | Overall information loss | ✓ |
| `height` | Generalization depth | ✓ |
| `precision` | Preserved granularity | ✓ |
| `nm_entropy` | Frequency-weighted entropy | ✓ |
| `entropy` | Information-theoretic entropy | ✗ |
| `discernibility` | Sum of squared group sizes | ✗ |
| `aecs` | Average equivalence class size | ✗ |
| `ambiguity` | Record ambiguity level | ✗ |
| `classification` | Classification utility | ✗ |

---

## Configuration

The quality model is specified in the JSON configuration file of each experiment:

```json
{
  "utility_measure": "loss",
  "utility_aggregate": "arithmetic_mean"
}
```

The default value is `loss` with the `arithmetic_mean` aggregate.

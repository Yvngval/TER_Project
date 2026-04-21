"""Sklearn pipelines for classification-based utility, mirroring ARX classifiers.

Each pipeline is designed to reproduce the preprocessing and model behaviour of
its ARX counterpart as closely as possible, so that classification accuracy on
the anonymized dataset can be compared directly with ARX output.

Classifiers
-----------
- **ZeroR**               → ``DummyClassifier(strategy="prior")``
  (ARX baseline: predicts class-prior probabilities)
- **Logistic Regression** → ``LogisticRegression``
  (ARX: Mahout ``OnlineLogisticRegression`` with ElasticNet regularisation)
- **Naive Bayes**         → ``MultinomialNB`` or ``BernoulliNB``
  (ARX: SMILE ``NaiveBayes``; numeric features are first discretized into bins)
- **Random Forest**       → ``RandomForestClassifier``
  (ARX: SMILE ``RandomForest``; no feature scaling, ordinal encoding for categoricals)

Design notes
------------
- Each model factory returns a *fresh* ``Pipeline`` instance so that no state
  is shared across cross-validation folds.
- Generalised interval values (e.g. ``"30-39"``) are converted to their numeric
  midpoint before any further preprocessing.
- Suppressed values (``"*"``) are treated as unknown categories or imputed via
  the most-frequent strategy, depending on the pipeline.
- Hyperparameters are merged at call time from ``classification_config.json``
  over built-in defaults that mirror ARX settings.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# ---------------------------------------------------------------------------
# Column metadata for the Adult dataset
# ---------------------------------------------------------------------------

ADULT_NUMERIC_COLS: list[str] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
"""Columns of the Adult dataset that carry continuous numeric values.

These are the only columns that receive numeric preprocessing (midpoint
conversion, scaling, binning).  All other feature columns are treated as
categorical regardless of their actual values in any given anonymized output.
"""

ADULT_FEATURE_COLS: list[str] = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]
"""Ordered list of all 14 feature columns used for classification.

The target column (``income``) is excluded.  This list matches the feature
order expected by ARX and must remain consistent with the column order in the
anonymized CSV outputs.
"""


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
# These values mirror ARX defaults.  They are used when a key is absent or
# only partially specified in classification_config.json, so the actual params
# passed to the classifier are always: {**DEFAULT_*_PARAMS, **cfg_params}.

DEFAULT_LR_PARAMS: dict = {
    "C":         100000,   # very weak L2 regularisation (≈ ARX no-regularisation default)
    "penalty":   "elasticnet",
    "l1_ratio":  0.5,      # equal mix of L1 and L2 (ARX ElasticNet default)
    "solver":    "saga",   # required for elasticnet penalty
    "max_iter":  2000,
}
"""Default hyperparameters for ``LogisticRegression``.

``C=100_000`` is intentionally large so that regularisation has almost no
effect, matching the unregularised ARX ``OnlineLogisticRegression``.
``solver="saga"`` is required by scikit-learn when ``penalty="elasticnet"``.
"""

DEFAULT_NB_PARAMS: dict = {
    "alpha": 1.0,
}
"""Default hyperparameters for Naive Bayes (Multinomial or Bernoulli).

``alpha=1.0`` is Laplace smoothing, matching the SMILE NaiveBayes default.
"""

DEFAULT_RF_PARAMS: dict = {
    "n_estimators":    500,
    "max_features":    "sqrt",   # ARX SMILE default: sqrt(n_features)
    "min_samples_leaf": 5,
    "max_leaf_nodes":  100,
    "bootstrap":       True,
    "max_samples":     1.0,      # use all samples per tree (no sub-sampling)
    "criterion":       "gini",
}
"""Default hyperparameters for ``RandomForestClassifier``.

Values match the ARX SMILE ``RandomForest`` defaults.  ``max_samples=1.0``
with ``bootstrap=True`` means each tree is trained on a bootstrap resample of
the full training set, exactly as ARX does.
"""


# ---------------------------------------------------------------------------
# Midpoint transformer
# ---------------------------------------------------------------------------

def _range_midpoint_scalar(value: str) -> float:
    """Convert a generalised value string to its float midpoint.

    Handles three cases produced by ARX generalisation hierarchies:

    - **Exact numeric** ``"37"`` → ``37.0``
    - **Range** ``"30-39"`` or ``"30–39"`` (en-dash) → ``34.5``
    - **Suppressed / unknown** ``"*"``, ``"?"``, ``""`` → ``NaN``

    Parameters
    ----------
    value : str
        A single cell value, as found in an anonymized CSV column.

    Returns
    -------
    float
        The numeric midpoint, or ``NaN`` when the value cannot be converted.

    Examples
    --------
    >>> _range_midpoint_scalar("30-39")
    34.5
    >>> _range_midpoint_scalar("37")
    37.0
    >>> import math; math.isnan(_range_midpoint_scalar("*"))
    True
    """
    value = str(value).strip()
    if value in ("*", "?", "", "nan"):
        return np.nan
    try:
        return float(value)
    except ValueError:
        pass
    m = re.match(r"^(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)$", value)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


# Vectorised version: applies _range_midpoint_scalar element-wise without a
# Python-level double for-loop (numpy dispatches the ufunc internally).
_midpoint_vec = np.vectorize(_range_midpoint_scalar, otypes=[float])


def _midpoint_array(X) -> np.ndarray:
    """Apply midpoint conversion element-wise to a 2-D array or DataFrame.

    This is the function passed to ``FunctionTransformer`` inside numeric
    preprocessing pipelines.  scikit-learn may pass either a ``numpy.ndarray``
    or a ``pandas.DataFrame`` depending on the input type, so the argument is
    coerced to ``object`` dtype before dispatching ``_midpoint_vec``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Raw string values (possibly containing ranges like ``"30-39"`` or
        suppressed values ``"*"``).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features), dtype float
        Numeric midpoints; suppressed / non-parseable values become ``NaN``.
    """
    return _midpoint_vec(np.asarray(X, dtype=object))


# ---------------------------------------------------------------------------
# Column split helper
# ---------------------------------------------------------------------------

def split_num_cat(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split a feature column list into numeric and categorical sub-lists.

    The split is based on membership in ``ADULT_NUMERIC_COLS``.  Columns that
    appear in the feature list but *not* in ``ADULT_NUMERIC_COLS`` are treated
    as categorical.  The relative order of columns within each sub-list is
    preserved.

    Parameters
    ----------
    feature_cols : list[str]
        Subset (or full set) of ``ADULT_FEATURE_COLS`` to split.

    Returns
    -------
    num_cols : list[str]
        Feature columns that are numeric (a subset of ``ADULT_NUMERIC_COLS``).
    cat_cols : list[str]
        Remaining feature columns, treated as categorical.

    Examples
    --------
    >>> num, cat = split_num_cat(["age", "workclass", "capital-gain"])
    >>> num
    ['age', 'capital-gain']
    >>> cat
    ['workclass']
    """
    num = [c for c in feature_cols if c in ADULT_NUMERIC_COLS]
    cat = [c for c in feature_cols if c not in ADULT_NUMERIC_COLS]
    return num, cat


# ---------------------------------------------------------------------------
# Internal preprocessor factories
# ---------------------------------------------------------------------------

def _make_numeric_pipe_scaled() -> Pipeline:
    """Build a numeric preprocessing pipeline with StandardScaler.

    Steps
    -----
    1. **midpoint** — convert range strings and suppressed values to floats
       (``"30-39"`` → ``34.5``, ``"*"`` → ``NaN``).
    2. **imputer** — replace ``NaN`` with the column mean.
    3. **scaler** — zero-mean, unit-variance normalisation.

    Used by: Logistic Regression (gradient-descent solver is sensitive to scale).

    Returns
    -------
    Pipeline
        A fitted-ready sklearn Pipeline for numeric columns.
    """
    return Pipeline([
        ("midpoint", FunctionTransformer(_midpoint_array, validate=False)),
        ("imputer",  SimpleImputer(strategy="mean")),
        ("scaler",   StandardScaler()),
    ])


def _make_numeric_pipe_plain() -> Pipeline:
    """Build a numeric preprocessing pipeline without scaling.

    Steps
    -----
    1. **midpoint** — convert range strings and suppressed values to floats.
    2. **imputer** — replace ``NaN`` with the column mean.

    Used by: Random Forest (decision trees are scale-invariant; scaling would
    add noise without benefit and would diverge from ARX behaviour).

    Returns
    -------
    Pipeline
        A fitted-ready sklearn Pipeline for numeric columns.
    """
    return Pipeline([
        ("midpoint", FunctionTransformer(_midpoint_array, validate=False)),
        ("imputer",  SimpleImputer(strategy="mean")),
    ])


def _make_numeric_pipe_binned(n_bins: int = 10,
                              bin_strategy: str = "uniform") -> Pipeline:
    """Build a numeric preprocessing pipeline with equal-width binning.

    Steps
    -----
    1. **midpoint** — convert range strings and suppressed values to floats.
    2. **imputer** — replace ``NaN`` with the column mean.
    3. **bins** — discretize into ``n_bins`` bins and one-hot encode the result
       (``encode="onehot-dense"`` avoids a separate ``OneHotEncoder`` step).

    Used by: Naive Bayes (requires non-negative, discrete features).

    Parameters
    ----------
    n_bins : int, default 10
        Number of equal-width bins per numeric feature.
    bin_strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy passed to ``KBinsDiscretizer``.
        ``"uniform"`` (equal-width) better matches ARX/SMILE behaviour on
        skewed features such as ``capital-gain`` where most values are 0:
        quantile binning would collapse the zero-mass into a single bin.

    Returns
    -------
    Pipeline
        A fitted-ready sklearn Pipeline for numeric columns.
    """
    return Pipeline([
        ("midpoint", FunctionTransformer(_midpoint_array, validate=False)),
        ("imputer",  SimpleImputer(strategy="mean")),
        ("bins",     KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy=bin_strategy,
            subsample=None,
        )),
    ])


def _make_categorical_pipe_onehot() -> Pipeline:
    """Build a categorical preprocessing pipeline with one-hot encoding.

    Steps
    -----
    1. **ordinal** — ``OrdinalEncoder`` with ``handle_unknown="use_encoded_value"``
       and ``unknown_value=-1`` so that suppressed cells (``"*"``) seen at
       transform time are mapped to ``-1`` instead of raising an error.
    2. **imputer** — replace ``-1`` (unknown) with the most-frequent category
       code via ``strategy="most_frequent"``.
    3. **onehot** — ``OneHotEncoder`` with ``handle_unknown="ignore"`` produces
       a dense matrix; unseen categories at test time become all-zeros.

    Used by: Logistic Regression and Naive Bayes.

    Returns
    -------
    Pipeline
        A fitted-ready sklearn Pipeline for categorical columns.
    """
    return Pipeline([
        ("ordinal",  OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("onehot",   OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def _make_categorical_pipe_ordinal() -> Pipeline:
    """Build a categorical preprocessing pipeline with ordinal encoding only.

    Steps
    -----
    1. **ordinal** — ``OrdinalEncoder`` with ``unknown_value=-1`` maps each
       category to an integer code; suppressed values (``"*"``) become ``-1``.
    2. **imputer** — replace ``-1`` codes with the most-frequent category code.

    Used by: Random Forest (trees split on threshold values so ordinal codes
    are sufficient; one-hot encoding would unnecessarily increase dimensionality
    and slow training without improving accuracy).

    Returns
    -------
    Pipeline
        A fitted-ready sklearn Pipeline for categorical columns.
    """
    return Pipeline([
        ("ordinal",  OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("imputer",  SimpleImputer(strategy="most_frequent")),
    ])


def _build_column_transformer(
    num_pipe: Pipeline | None,
    cat_pipe: Pipeline | None,
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """Assemble a ``ColumnTransformer`` from numeric and categorical pipelines.

    Only non-empty column lists are registered; if a list is empty the
    corresponding pipe is silently skipped.  ``remainder="drop"`` ensures
    that any column not in either list is discarded.

    Parameters
    ----------
    num_pipe : Pipeline or None
        Preprocessing pipeline for numeric columns.  Omitted when ``num_cols``
        is empty or ``None`` is passed.
    cat_pipe : Pipeline or None
        Preprocessing pipeline for categorical columns.  Omitted when
        ``cat_cols`` is empty or ``None`` is passed.
    num_cols : list[str]
        Names of numeric columns to route through ``num_pipe``.
    cat_cols : list[str]
        Names of categorical columns to route through ``cat_pipe``.

    Returns
    -------
    ColumnTransformer
        Ready to be embedded as the ``"preprocessor"`` step of a ``Pipeline``.
    """
    transformers: list = []
    if num_cols and num_pipe is not None:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols and cat_pipe is not None:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_zero_r() -> DummyClassifier:
    """Build the ZeroR baseline classifier (mirrors ARX ZeroR).

    ``strategy="prior"`` makes ``predict_proba`` return the empirical class
    frequencies estimated from the training fold, rather than a constant
    ``[1.0, 0.0]`` vector.  As a result, the out-of-fold probabilities vary
    slightly across folds (because each fold has different class frequencies),
    so ``roc_auc_score`` on the OOF predictions deviates slightly from the
    theoretical 0.5 — matching the small deviations ARX reports (e.g. 0.49).
    ``predict()`` still returns the majority class, so accuracy equals the
    majority-class baseline.

    Returns
    -------
    DummyClassifier
        An unfitted ``DummyClassifier(strategy="prior")``.
    """
    return DummyClassifier(strategy="prior")


def build_logistic_regression(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> Pipeline:
    """Build a Logistic Regression pipeline (mirrors ARX Mahout OnlineLogisticRegression).

    Preprocessing
    -------------
    - **Numeric** : midpoint conversion → mean imputation → ``StandardScaler``
    - **Categorical** : ``OrdinalEncoder`` → most-frequent imputation → ``OneHotEncoder``

    The default ``C=100_000`` and ``penalty="elasticnet"`` (L1/L2 mix) match
    the ARX ``OnlineLogisticRegression`` configuration.

    Parameters
    ----------
    num_cols : list[str]
        Names of numeric feature columns.
    cat_cols : list[str]
        Names of categorical feature columns.
    params : dict
        ``LogisticRegression`` keyword arguments.  Typically a merge of
        ``DEFAULT_LR_PARAMS`` with any overrides from ``classification_config.json``.

    Returns
    -------
    Pipeline
        An unfitted sklearn Pipeline with steps ``["preprocessor", "classifier"]``.
    """
    preprocessor = _build_column_transformer(
        _make_numeric_pipe_scaled(),
        _make_categorical_pipe_onehot(),
        num_cols,
        cat_cols,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(**params)),
    ])


def build_naive_bayes(
    num_cols: list[str],
    cat_cols: list[str],
    model: str,
    params: dict,
    n_bins: int = 10,
    bin_strategy: str = "uniform",
) -> Pipeline:
    """Build a Naive Bayes pipeline (mirrors ARX SMILE NaiveBayes).

    Numeric features are discretized into bins and one-hot encoded before
    being fed to the classifier, which requires non-negative discrete input.

    Preprocessing
    -------------
    - **Numeric** : midpoint conversion → mean imputation → ``KBinsDiscretizer``
      (with ``encode="onehot-dense"``)
    - **Categorical** : ``OrdinalEncoder`` → most-frequent imputation → ``OneHotEncoder``

    Parameters
    ----------
    num_cols : list[str]
        Names of numeric feature columns.
    cat_cols : list[str]
        Names of categorical feature columns.
    model : {"MULTINOMIAL", "BERNOULLI"}
        Which Naive Bayes variant to use.
        ``"MULTINOMIAL"`` → ``MultinomialNB`` (ARX/SMILE default).
        ``"BERNOULLI"``   → ``BernoulliNB``.
    params : dict
        Naive Bayes keyword arguments, typically merged from ``DEFAULT_NB_PARAMS``.
    n_bins : int, default 10
        Number of equal-width bins for numeric discretization.
    bin_strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy; ``"uniform"`` matches ARX/SMILE behaviour.

    Returns
    -------
    Pipeline
        An unfitted sklearn Pipeline with steps ``["preprocessor", "classifier"]``.
    """
    preprocessor = _build_column_transformer(
        _make_numeric_pipe_binned(n_bins, bin_strategy),
        _make_categorical_pipe_onehot(),
        num_cols,
        cat_cols,
    )
    clf = MultinomialNB(**params) if model.upper() == "MULTINOMIAL" else BernoulliNB(**params)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])


def build_random_forest(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict,
    random_state: int = 42,
) -> Pipeline:
    """Build a Random Forest pipeline (mirrors ARX SMILE RandomForest).

    Preprocessing
    -------------
    - **Numeric** : midpoint conversion → mean imputation (no scaling — decision
      trees split on thresholds and are scale-invariant)
    - **Categorical** : ``OrdinalEncoder`` → most-frequent imputation (ordinal
      codes are sufficient; one-hot encoding is unnecessary for tree-based models)

    Parameters
    ----------
    num_cols : list[str]
        Names of numeric feature columns.
    cat_cols : list[str]
        Names of categorical feature columns.
    params : dict
        ``RandomForestClassifier`` keyword arguments, typically merged from
        ``DEFAULT_RF_PARAMS``.  ``random_state`` and ``n_jobs`` are set
        separately and should not appear in ``params``.
    random_state : int, default 42
        Random seed for reproducibility, forwarded to ``RandomForestClassifier``.

    Returns
    -------
    Pipeline
        An unfitted sklearn Pipeline with steps ``["preprocessor", "classifier"]``.
        The classifier is built with ``n_jobs=-1`` to use all available cores.
    """
    preprocessor = _build_column_transformer(
        _make_numeric_pipe_plain(),
        _make_categorical_pipe_ordinal(),
        num_cols,
        cat_cols,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_model_builders(
    num_cols: list[str],
    cat_cols: list[str],
    config: dict,
    random_state: int = 42,
) -> dict[str, callable]:
    """Return a registry of zero-argument model factories.

    Each factory, when called with no arguments, produces a fresh unfitted
    ``Pipeline`` (or ``DummyClassifier`` for ZeroR).  Because factories are
    called once per cross-validation fold, no state is shared between folds.

    Only models listed in ``config["active_models"]`` are included.  When
    ``active_models`` is ``"all"`` (the default), all four classifiers are
    registered.

    Hyperparameters are resolved by merging the built-in defaults with any
    overrides found under ``config["classifiers"][<name>]["params"]``.  Keys
    present in the config override defaults; absent keys fall back to defaults.

    Parameters
    ----------
    num_cols : list[str]
        Names of numeric feature columns (passed to each builder).
    cat_cols : list[str]
        Names of categorical feature columns (passed to each builder).
    config : dict
        Full contents of ``classification_config.json``.  Expected structure::

            {
              "active_models": "all" | ["zero_r", "logistic_regression", ...],
              "classifiers": {
                "logistic_regression": {"params": {...}},
                "naive_bayes": {
                  "model": "MULTINOMIAL",
                  "params": {...},
                  "preprocessing": {"n_bins": 10, "bin_strategy": "uniform"}
                },
                "random_forest": {"params": {...}}
              }
            }

    random_state : int, default 42
        Passed to ``build_random_forest`` for reproducibility.

    Returns
    -------
    dict[str, callable]
        Mapping of model name (e.g. ``"logistic_regression"``) to a
        zero-argument callable that returns a fresh unfitted pipeline.

    Examples
    --------
    >>> builders = get_model_builders(num_cols, cat_cols, config)
    >>> pipeline = builders["logistic_regression"]()
    >>> pipeline.fit(X_train, y_train)
    """
    classifiers_cfg = config.get("classifiers", {})
    builders: dict[str, callable] = {}

    active = config.get("active_models", "all")
    all_models = {"zero_r", "logistic_regression", "naive_bayes", "random_forest"}
    if active == "all":
        enabled_models = all_models
    else:
        enabled_models = {m.strip() for m in active} if not isinstance(active, str) else {active.strip()}

    # ZeroR
    if "zero_r" in enabled_models:
        builders["zero_r"] = build_zero_r

    # Logistic Regression
    lr_cfg = classifiers_cfg.get("logistic_regression", {})
    if "logistic_regression" in enabled_models:
        lr_params = {**DEFAULT_LR_PARAMS, **lr_cfg.get("params", {})}
        builders["logistic_regression"] = lambda p=lr_params: build_logistic_regression(
            num_cols, cat_cols, p
        )

    # Naive Bayes
    nb_cfg = classifiers_cfg.get("naive_bayes", {})
    if "naive_bayes" in enabled_models:
        nb_model     = nb_cfg.get("model", "MULTINOMIAL")
        nb_params    = {**DEFAULT_NB_PARAMS, **nb_cfg.get("params", {})}
        nb_n_bins    = nb_cfg.get("preprocessing", {}).get("n_bins", 10)
        nb_bin_strat = nb_cfg.get("preprocessing", {}).get("bin_strategy", "uniform")
        builders["naive_bayes"] = lambda m=nb_model, p=nb_params, b=nb_n_bins, s=nb_bin_strat: build_naive_bayes(
            num_cols, cat_cols, m, p, b, s
        )

    # Random Forest
    rf_cfg = classifiers_cfg.get("random_forest", {})
    if "random_forest" in enabled_models:
        rf_params = {**DEFAULT_RF_PARAMS, **rf_cfg.get("params", {})}
        builders["random_forest"] = lambda p=rf_params: build_random_forest(
            num_cols, cat_cols, p, random_state
        )

    return builders

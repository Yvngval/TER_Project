# Statistical utility metrics for comparing original vs anonymized datasets.
# Metrics: KL divergence, Total Variation Distance, Wasserstein distance,
# mean/std delta, and correlation matrix delta.

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _range_midpoint(value: str) -> float | None:
    """Return the numeric midpoint of a generalized range string.

    Examples:
        "35-39"  -> 37.0
        "25"     -> 25.0
        "*"      -> None  (fully suppressed / wildcard)
    """
    value = value.strip()
    if value in ("*", "?", "", "nan"):
        return None
    # Exact numeric value
    try:
        return float(value)
    except ValueError:
        pass
    # Range like "35-39" or "35–39" (en-dash)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)$", value)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return None


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series that may contain range strings to float via midpoints."""
    return series.astype(str).map(_range_midpoint)


def _try_numeric_conversion(
    series: pd.Series,
    direct_threshold: float = 0.5,
    midpoint_threshold: float = 0.5,
) -> pd.Series | None:
    """Try to convert a series to numeric values, using midpoints for intervals.

    Returns the numeric series if more than the given threshold of values
    are convertible, otherwise returns None (indicating a purely categorical column).
    """
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > direct_threshold:
        return num
    mid = _to_numeric_series(series)
    if mid.notna().mean() > midpoint_threshold:
        return mid
    return None


# ---------------------------------------------------------------------------
# Per-column metrics
# ---------------------------------------------------------------------------

def kl_divergence(
    orig: pd.Series, anon: pd.Series, smoothing: float = 1e-10, n_bins: int = 20
) -> float:
    """KL divergence D(P || Q) between the two series.

    For numeric or interval-encoded columns, both series are converted to
    numeric values (via midpoints for intervals) and discretized into common
    bins before computing the divergence.  For purely categorical columns,
    the original category-based comparison is used.

    Parameters
    ----------
    n_bins : int
        Number of bins for discretizing numeric columns (default 20).
    """
    # Detect numeric / interval columns
    orig_num = _try_numeric_conversion(orig)
    anon_num = _try_numeric_conversion(anon)

    if orig_num is not None and anon_num is not None:
        # Numeric path: discretize into common bins derived from original data
        orig_clean = orig_num.dropna().values
        anon_clean = anon_num.dropna().values
        if len(orig_clean) > 0 and len(anon_clean) > 0:
            bin_edges = np.histogram_bin_edges(orig_clean, bins=n_bins)
            p = np.histogram(orig_clean, bins=bin_edges)[0].astype(float) + smoothing
            q = np.histogram(anon_clean, bins=bin_edges)[0].astype(float) + smoothing
            p /= p.sum()
            q /= q.sum()
            return float(np.sum(p * np.log(np.clip(p, 1e-300, None) / np.clip(q, 1e-300, None))))

    # Categorical path (original behavior)
    p_counts = orig.astype(str).value_counts()
    q_counts = anon.astype(str).value_counts()

    all_cats = p_counts.index.union(q_counts.index)

    p = np.array([p_counts.get(c, 0) for c in all_cats], dtype=float) + smoothing
    q = np.array([q_counts.get(c, 0) for c in all_cats], dtype=float) + smoothing
    p /= p.sum()
    q /= q.sum()

    # KL divergence: sum(p * log(p/q)), clip to avoid log(0)
    return float(np.sum(p * np.log(np.clip(p, 1e-300, None) / np.clip(q, 1e-300, None))))


def total_variation_distance(
    orig: pd.Series, anon: pd.Series, n_bins: int = 20
) -> float:
    """TVD = 0.5 * sum |P(x) - Q(x)|. Range: [0, 1].

    For numeric or interval-encoded columns, both series are discretized
    into common bins before comparing distributions.  For purely categorical
    columns, the original category-based comparison is used.

    Parameters
    ----------
    n_bins : int
        Number of bins for discretizing numeric columns (default 20).
    """
    # Detect numeric / interval columns
    orig_num = _try_numeric_conversion(orig)
    anon_num = _try_numeric_conversion(anon)

    if orig_num is not None and anon_num is not None:
        # Numeric path: discretize into common bins derived from original data
        orig_clean = orig_num.dropna().values
        anon_clean = anon_num.dropna().values
        if len(orig_clean) > 0 and len(anon_clean) > 0:
            bin_edges = np.histogram_bin_edges(orig_clean, bins=n_bins)
            p = np.histogram(orig_clean, bins=bin_edges)[0].astype(float)
            q = np.histogram(anon_clean, bins=bin_edges)[0].astype(float)
            p_total = p.sum()
            q_total = q.sum()
            if p_total > 0 and q_total > 0:
                p /= p_total
                q /= q_total
                return float(0.5 * np.abs(p - q).sum())

    # Categorical path (original behavior)
    p = orig.astype(str).value_counts(normalize=True)
    q = anon.astype(str).value_counts(normalize=True)

    all_cats = p.index.union(q.index)
    p = p.reindex(all_cats, fill_value=0.0)
    q = q.reindex(all_cats, fill_value=0.0)

    return float(0.5 * np.abs(p.values - q.values).sum())


def wasserstein_dist(orig: pd.Series, anon: pd.Series) -> float | None:
    """Wasserstein-1 distance for numeric or range-encoded columns.

    Returns None if the column cannot be converted to numeric.
    """
    # Try direct numeric conversion first
    orig_num = pd.to_numeric(orig, errors="coerce")
    anon_num = pd.to_numeric(anon, errors="coerce")

    # Fall back to midpoint extraction for generalized ranges
    if orig_num.isna().mean() > 0.5:
        orig_num = _to_numeric_series(orig)
    if anon_num.isna().mean() > 0.5:
        anon_num = _to_numeric_series(anon)

    orig_clean = orig_num.dropna().values
    anon_clean = anon_num.dropna().values

    if len(orig_clean) == 0 or len(anon_clean) == 0:
        return None

    # Wasserstein-1 via sorted CDFs (equivalent to scipy's wasserstein_distance)
    orig_sorted = np.sort(orig_clean)
    anon_sorted = np.sort(anon_clean)
    # Interpolate to common size
    n = max(len(orig_sorted), len(anon_sorted))
    orig_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(orig_sorted)), orig_sorted)
    anon_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(anon_sorted)), anon_sorted)
    return float(np.mean(np.abs(orig_interp - anon_interp)))


def mean_std_delta(orig: pd.Series, anon: pd.Series) -> dict[str, float] | None:
    """Absolute difference in mean and standard deviation for numeric-convertible columns."""
    orig_num = pd.to_numeric(orig, errors="coerce")
    anon_num = pd.to_numeric(anon, errors="coerce")

    if orig_num.isna().mean() > 0.5:
        orig_num = _to_numeric_series(orig)
    if anon_num.isna().mean() > 0.5:
        anon_num = _to_numeric_series(anon)

    orig_clean = orig_num.dropna()
    anon_clean = anon_num.dropna()

    if len(orig_clean) == 0 or len(anon_clean) == 0:
        return None

    mean_orig = float(orig_clean.mean())
    mean_anon = float(anon_clean.mean())
    mean_delta = float(abs(mean_orig - mean_anon))
    std_orig = float(orig_clean.std())
    std_anon = float(anon_clean.std())
    std_delta = float(abs(std_orig - std_anon))

    return {
        "mean_orig": mean_orig,
        "mean_anon": mean_anon,
        "mean_delta": mean_delta,
        "mean_delta_relative": mean_delta / abs(mean_orig) if mean_orig != 0 else None,
        "std_orig": std_orig,
        "std_anon": std_anon,
        "std_delta": std_delta,
        "std_delta_relative": std_delta / std_orig if std_orig != 0 else None,
    }


# ---------------------------------------------------------------------------
# Global structural metric
# ---------------------------------------------------------------------------

def _encode_for_correlation(
    orig_df: pd.DataFrame, anon_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode two dataframes for correlation computation with consistent encoding.

    - Numeric columns: kept as-is.
    - Range-string columns (e.g. "35-39"): converted to midpoints.
    - Categorical columns: label-encoded using a common CategoricalDtype
      so that identical labels receive identical codes in both dataframes.

    Note: Pearson correlation on label-encoded nominal variables (e.g. race,
    native-country) is an approximation.  A more rigorous alternative for
    nominal variables would be Cramér's V.
    """
    orig_result = pd.DataFrame(index=orig_df.index)
    anon_result = pd.DataFrame(index=anon_df.index)

    common_cols = [c for c in orig_df.columns if c in anon_df.columns]

    for col in common_cols:
        orig_s = orig_df[col]
        anon_s = anon_df[col]

        # Try direct numeric conversion on both series
        orig_num = pd.to_numeric(orig_s, errors="coerce")
        anon_num = pd.to_numeric(anon_s, errors="coerce")
        orig_num_ok = orig_num.notna().mean() > 0.5
        anon_num_ok = anon_num.notna().mean() > 0.5

        if orig_num_ok and anon_num_ok:
            orig_result[col] = orig_num
            anon_result[col] = anon_num
            continue

        # Try midpoint conversion for interval ranges
        orig_mid = _to_numeric_series(orig_s)
        anon_mid = _to_numeric_series(anon_s)

        # Pick the best numeric representation for each series independently
        orig_best = orig_num if orig_num_ok else (orig_mid if orig_mid.notna().mean() > 0.3 else None)
        anon_best = anon_num if anon_num_ok else (anon_mid if anon_mid.notna().mean() > 0.3 else None)

        if orig_best is not None and anon_best is not None:
            orig_result[col] = orig_best
            anon_result[col] = anon_best
            continue

        # Label encode with common CategoricalDtype for consistent codes.
        # Note: Pearson correlation on nominal variables is an approximation;
        # Cramér's V would be a more rigorous alternative.
        all_categories = sorted(
            set(orig_s.astype(str).unique()) | set(anon_s.astype(str).unique())
        )
        common_dtype = pd.CategoricalDtype(categories=all_categories, ordered=False)
        orig_result[col] = (
            orig_s.astype(str).astype(common_dtype).cat.codes.replace(-1, np.nan)
        )
        anon_result[col] = (
            anon_s.astype(str).astype(common_dtype).cat.codes.replace(-1, np.nan)
        )

    return orig_result, anon_result


def correlation_matrix_delta(orig_df: pd.DataFrame, anon_df: pd.DataFrame) -> dict:
    """Frobenius norm of the difference between the two correlation matrices.

    A value of 0 means perfect preservation of linear relationships.
    Normalized by 2 * sqrt(n * (n - 1)), the theoretical maximum for the
    Frobenius norm of the difference between two correlation matrices
    (diagonal elements are always 0, off-diagonal elements are in [-2, 2]).
    """
    orig_enc, anon_enc = _encode_for_correlation(orig_df, anon_df)
    orig_enc = orig_enc.dropna(axis=1, how="all")
    anon_enc = anon_enc.dropna(axis=1, how="all")

    common_cols = [c for c in orig_enc.columns if c in anon_enc.columns]
    if len(common_cols) < 2:
        return {"frobenius_norm": None, "frobenius_norm_normalized": None, "n_columns": len(common_cols)}

    corr_orig = orig_enc[common_cols].corr().fillna(0).values
    corr_anon = anon_enc[common_cols].corr().fillna(0).values

    diff = corr_orig - corr_anon
    frobenius = float(np.linalg.norm(diff, "fro"))
    n = len(common_cols)
    max_possible = float(2 * np.sqrt(n * (n - 1)))

    return {
        "frobenius_norm": frobenius,
        "frobenius_norm_normalized": frobenius / max_possible if max_possible > 0 else None,
        "n_columns": n,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_column_metrics(orig_df: pd.DataFrame, anon_df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Compute per-column utility metrics for the given list of columns."""
    metrics: dict[str, dict] = {}
    for col in columns:
        if col not in orig_df.columns or col not in anon_df.columns:
            continue

        orig_col = orig_df[col]
        anon_col = anon_df[col]

        col_metrics: dict = {
            "kl_divergence": kl_divergence(orig_col, anon_col),
            "total_variation_distance": total_variation_distance(orig_col, anon_col),
        }

        w = wasserstein_dist(orig_col, anon_col)
        if w is not None:
            col_metrics["wasserstein_distance"] = w
            # Normalized Wasserstein: divide by std of original numeric values
            orig_num = pd.to_numeric(orig_col, errors="coerce")
            if orig_num.isna().mean() > 0.5:
                orig_num = _to_numeric_series(orig_col)
            orig_std = float(np.std(orig_num.dropna().values))
            col_metrics["wasserstein_distance_normalized"] = (
                w / orig_std if orig_std > 0 else None
            )

        ms = mean_std_delta(orig_col, anon_col)
        if ms is not None:
            col_metrics.update(ms)

        metrics[col] = col_metrics
    return metrics


def compute_utility_metrics(
    orig_df: pd.DataFrame,
    anon_df: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_attributes: list[str],
) -> dict:
    """Compute all statistical utility metrics between original and anonymized datasets.

    Returns a dict with:
    - Record counts and suppression rate
    - Per-column metrics (KL, TVD, Wasserstein, mean/std delta)
    - Correlation matrix delta (Frobenius norm)
    - Aggregate scores (mean TVD/KL/Wasserstein across QIs and sensitive attributes)
    """
    all_columns = quasi_identifiers + sensitive_attributes

    col_metrics = compute_column_metrics(orig_df, anon_df, all_columns)
    corr_delta = correlation_matrix_delta(orig_df, anon_df)

    qi_tvds = [col_metrics[c]["total_variation_distance"] for c in quasi_identifiers if c in col_metrics]
    sensitive_tvds = [col_metrics[c]["total_variation_distance"] for c in sensitive_attributes if c in col_metrics]
    qi_kls = [col_metrics[c]["kl_divergence"] for c in quasi_identifiers if c in col_metrics]

    qi_ws = [
        col_metrics[c]["wasserstein_distance"]
        for c in quasi_identifiers
        if c in col_metrics and "wasserstein_distance" in col_metrics[c]
    ]
    all_ws = [
        col_metrics[c]["wasserstein_distance"]
        for c in all_columns
        if c in col_metrics and "wasserstein_distance" in col_metrics[c]
    ]

    return {
        "n_records_orig": len(orig_df),
        "n_records_anon": len(anon_df),
        "suppression_rate": round(1.0 - len(anon_df) / len(orig_df), 6) if len(orig_df) > 0 else None,
        "per_column": col_metrics,
        "correlation_delta": corr_delta,
        "mean_tvd_qi": float(np.mean(qi_tvds)) if qi_tvds else None,
        "mean_tvd_sensitive": float(np.mean(sensitive_tvds)) if sensitive_tvds else None,
        "mean_tvd_all": float(np.mean(qi_tvds + sensitive_tvds)) if (qi_tvds + sensitive_tvds) else None,
        "mean_kl_qi": float(np.mean(qi_kls)) if qi_kls else None,
        "mean_wasserstein_qi": float(np.mean(qi_ws)) if qi_ws else None,
        "mean_wasserstein_all": float(np.mean(all_ws)) if all_ws else None,
    }

"""Healthy-baseline anomaly detection for FRA feature vectors using IsolationForest."""

from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

DEFAULT_ANOMALY_MODEL_PATH = "models/isolation_forest.pkl"


def _as_feature_matrix(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def train_anomaly_detector(
    healthy_features: Any,
    *,
    random_state: int = 42,
    contamination: float = 0.05,
    n_estimators: int = 100,
    **kwargs: Any,
) -> IsolationForest:
    """
    Fit an :class:`~sklearn.ensemble.IsolationForest` **only** on feature rows from healthy FRA runs.

    The forest learns the healthy feature distribution; points far from that bulk in new data
    tend to receive lower scores and can be flagged as anomalies.

    Parameters
    ----------
    healthy_features
        ``(n_healthy, n_features)`` array or list of rows. At least two healthy samples
        are required for a stable fit.
    random_state
        RNG seed for the ensemble.
    contamination
        Upper bound on the fraction of outliers assumed in **training** data. Use a small
        value (e.g. ``0.01``–``0.1``) when the training set is almost entirely healthy.
    n_estimators
        Number of isolation trees.
    **kwargs
        Forwarded to :class:`~sklearn.ensemble.IsolationForest`.

    Returns
    -------
    IsolationForest
        Fitted estimator.
    """
    X = _as_feature_matrix(healthy_features)
    if X.shape[0] < 2:
        raise ValueError(
            "Need at least two healthy samples (rows) to train IsolationForest; "
            f"got {X.shape[0]}."
        )

    model = IsolationForest(
        n_estimators=n_estimators,
        random_state=random_state,
        contamination=contamination,
        **kwargs,
    )
    model.fit(X)
    return model


def predict_anomaly(
    features: Any,
    model: IsolationForest,
) -> dict[str, Any]:
    """
    Score a single new measurement in the same feature space used at training time.

    Parameters
    ----------
    features
        One feature vector (1-D) or a single row ``(1, n_features)``.
    model
        Fitted :class:`~sklearn.ensemble.IsolationForest` from :func:`train_anomaly_detector`.

    Returns
    -------
    dict
        ``is_anomaly`` — ``True`` if the forest labels the point as an outlier (``predict == -1``);
        ``score`` — :meth:`~sklearn.ensemble.IsolationForest.score_samples` value (higher means
        more *normal* / inlier-like; lower means more *abnormal* in sklearn’s convention).

    Raises
    ------
    ValueError
        If feature dimensionality does not match the model.
    """
    X = _as_feature_matrix(features)
    if X.shape[0] != 1:
        raise ValueError("Pass exactly one sample; use a 1-D feature vector or a single row.")
    if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature count {X.shape[1]} does not match model ({model.n_features_in_})."
        )

    label = int(model.predict(X)[0])
    score = float(model.score_samples(X)[0])
    return {
        "is_anomaly": label == -1,
        "score": score,
    }


def save_anomaly_model(model: IsolationForest, path: Optional[str] = None) -> str:
    """Persist a trained isolation forest with ``joblib``."""
    out = path or DEFAULT_ANOMALY_MODEL_PATH
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    joblib.dump(model, out)
    return out


def load_anomaly_model(path: Optional[str] = None) -> IsolationForest:
    """Load a model saved by :func:`save_anomaly_model`."""
    p = path or DEFAULT_ANOMALY_MODEL_PATH
    return joblib.load(p)


def _default_training_matrix() -> np.ndarray:
    """
    Synthetic paired-curve feature rows (mean_diff, std_diff, max_diff, correlation).

    Mirrors :func:`src.features.feature_extractor.extract_features` layout for a healthy
    envelope plus a few outlier-like rows.
    """
    return np.array(
        [
            [0.04, 0.02, 0.12, 0.995],
            [0.06, 0.03, 0.15, 0.992],
            [0.08, 0.04, 0.18, 0.988],
            [0.11, 0.06, 0.22, 0.982],
            [2.1, 1.2, 6.5, 0.72],
            [3.5, 2.0, 9.0, 0.55],
            [1.4, 0.9, 4.2, 0.84],
            [4.2, 2.6, 12.0, 0.48],
        ],
        dtype=np.float64,
    )


def ensure_anomaly_model(path: Optional[str] = None) -> IsolationForest:
    """
    Return a trained :class:`~sklearn.ensemble.IsolationForest` for 4-D paired features.

    If no file exists at ``path``, fits a small default ensemble on synthetic healthy/outlier
    rows and persists it next to the classifier artifacts.
    """
    p = path or DEFAULT_ANOMALY_MODEL_PATH
    if os.path.isfile(p):
        return load_anomaly_model(p)
    X = _default_training_matrix()
    model = train_anomaly_detector(X, contamination=0.25, random_state=42)
    save_anomaly_model(model, p)
    return model


def score_to_anomaly_0_100(score: float) -> float:
    """
    Map sklearn ``score_samples`` output to a 0–100 index where higher means more abnormal.

    IsolationForest ``score_samples``: higher values indicate more inlier-like / normal points.
    """
    # Empirical squashing for typical score ranges; keeps UI stable without calibration data.
    x = float(score)
    return float(np.clip(100.0 * (0.45 - x), 0.0, 100.0))

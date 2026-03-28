"""Random Forest training pipeline for FRA fault classification from feature matrices."""

from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Default artifact path (project root when cwd is FRA_AI_Data)
DEFAULT_MODEL_PATH = "models/trained_model.pkl"


def _as_feature_matrix(X: Any) -> np.ndarray:
    """Convert inputs to a 2-D float array."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _unique_labels(y: np.ndarray) -> list:
    """Sorted class labels for stable confusion matrix ordering."""
    lab = np.asarray(y)
    return sorted(np.unique(lab), key=lambda x: (str(type(x)), str(x)))


def train_model(
    X: Any,
    y: Any,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: Optional[str] = None,
    n_estimators: int = 100,
    **rf_kwargs: Any,
) -> dict[str, Any]:
    """
    Train a :class:`~sklearn.ensemble.RandomForestClassifier` on extracted FRA features.

    Performs a train/test split, fits the model, evaluates accuracy on the hold-out set,
    builds a confusion matrix, and persists the trained estimator with ``joblib``.

    Parameters
    ----------
    X
        Feature matrix ``(n_samples, n_features)`` or list of rows.
    y
        Fault labels (e.g. ``\"Healthy\"``, ``\"Winding Deformation\"``), length ``n_samples``.
    test_size
        Fraction of data for the test set (``0 < test_size < 1``). If there are too few
        samples to split, the full set is used for training and evaluation is done in-sample
        (see return dict flags).
    random_state
        RNG seed for splitting and the Random Forest.
    model_path
        Where to save the model; defaults to :data:`DEFAULT_MODEL_PATH`.
    n_estimators
        Number of trees in the forest.
    **rf_kwargs
        Extra arguments forwarded to :class:`~sklearn.ensemble.RandomForestClassifier`.

    Returns
    -------
    dict
        ``model`` — fitted classifier; ``test_accuracy`` — hold-out accuracy or in-sample
        accuracy when split is skipped; ``confusion_matrix`` — 2-D array;
        ``labels`` — row/column order for the matrix; ``model_path`` — save path;
        ``y_test``, ``y_pred`` — true/predicted labels used for evaluation;
        ``train_accuracy`` — accuracy on the training partition;
        ``used_train_test_split`` — whether a hold-out split was applied.
    """
    X_arr = _as_feature_matrix(X)
    y_arr = np.asarray(y)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples; got {X_arr.shape[0]} and {y_arr.shape[0]}."
        )
    if X_arr.shape[0] < 2:
        raise ValueError("Need at least 2 samples to train.")

    out_path = model_path or DEFAULT_MODEL_PATH
    labels_order = _unique_labels(y_arr)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        **rf_kwargs,
    )

    used_split = True
    min_test = max(1, int(round(X_arr.shape[0] * test_size)))
    if X_arr.shape[0] <= 2 or min_test >= X_arr.shape[0]:
        used_split = False

    if used_split:
        stratify = y_arr if _can_stratify(y_arr, test_size) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr,
                y_arr,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr,
                y_arr,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
    else:
        X_train, y_train = X_arr, y_arr
        X_test, y_test = X_arr, y_arr

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, y_train_pred))

    y_pred = clf.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    dir_name = os.path.dirname(out_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump(clf, out_path)

    return {
        "model": clf,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm,
        "labels": labels_order,
        "model_path": out_path,
        "y_test": y_test,
        "y_pred": y_pred,
        "used_train_test_split": used_split,
    }


def _can_stratify(y: np.ndarray, test_size: float) -> bool:
    """Stratified split needs at least two samples per class present."""
    _, counts = np.unique(y, return_counts=True)
    if np.any(counts < 2):
        return False
    n = y.shape[0]
    n_test = max(1, int(round(n * test_size)))
    n_train = n - n_test
    if n_train < 1:
        return False
    return True

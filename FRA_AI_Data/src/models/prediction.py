"""Fault classification from extracted features or paired FRA curves."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.features.feature_extractor import extract_features
from src.models.train_model import MODEL_PATH, load_model

FraInput = Union[pd.DataFrame, Mapping[str, Any]]


def _load_classifier(model_path: Optional[str]) -> Any:
    """Load a trained estimator from disk or the default bundle."""
    path = model_path or MODEL_PATH
    if model_path is None:
        return load_model()
    return joblib.load(path)


def _coerce_fra_frame(obj: FraInput) -> pd.DataFrame:
    """Build a DataFrame with ``Frequency`` and ``Magnitude`` columns."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, Mapping):
        freq = obj.get("Frequency")
        mag = obj.get("Magnitude")
        if freq is None:
            freq = obj.get("frequency")
        if mag is None:
            mag = obj.get("magnitude")
        if freq is None or mag is None:
            raise ValueError(
                "Expected keys 'Frequency'/'Magnitude' or 'frequency'/'magnitude'."
            )
        return pd.DataFrame(
            {
                "Frequency": np.asarray(freq, dtype=np.float64).ravel(),
                "Magnitude": np.asarray(mag, dtype=np.float64).ravel(),
            }
        )
    raise TypeError(f"Unsupported FRA input type: {type(obj)!r}")


def predict_from_features(
    features: Any,
    *,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the Random Forest on a single feature vector produced by your extraction pipeline.

    Parameters
    ----------
    features
        1-D sequence of feature values (one sample), e.g. from paired-curve
        :func:`src.features.feature_extractor.extract_features` or a flattened vector.
    model_path
        Optional path to a ``joblib``-saved model; defaults to the project model.

    Returns
    -------
    dict
        ``{\"fault\": str, \"confidence\": float}`` with confidence in ``[0, 1]``
        (maximum predicted class probability).
    """
    X = np.asarray(features, dtype=np.float64).ravel().reshape(1, -1)
    model = _load_classifier(model_path)

    fault = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    return {
        "fault": str(fault),
        "confidence": confidence,
    }


def predict_from_fra_pair(
    reference: FraInput,
    test: FraInput,
    *,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Extract paired-curve features (reference vs new signal) and classify the fault.

    Parameters
    ----------
    reference
        Baseline FRA curve: ``DataFrame`` with ``Frequency`` / ``Magnitude``, or a mapping
        from :func:`src.parser.parse_fra_file` (``frequency`` / ``magnitude`` arrays).
    test
        New FRA measurement in the same formats as ``reference``.
    model_path
        Optional ``joblib`` model path; defaults to the project model.

    Returns
    -------
    dict
        ``{\"fault\": str, \"confidence\": float}`` with confidence in ``[0, 1]``.
    """
    ref_df = _coerce_fra_frame(reference)
    test_df = _coerce_fra_frame(test)
    feats = extract_features(ref_df, test_df)
    return predict_from_features(feats, model_path=model_path)

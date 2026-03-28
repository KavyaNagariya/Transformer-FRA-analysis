"""Model path, loading, and bundled demo training for the FRA classifier."""

import os

import joblib

from src.models.training_pipeline import DEFAULT_MODEL_PATH, train_model as train_classifier

MODEL_PATH = DEFAULT_MODEL_PATH


def train_demo_model() -> dict:
    """
    Fit the default Random Forest on bundled 4-feature examples and save to ``MODEL_PATH``.

    Used when no saved model exists or loading fails. Feature order matches
    :func:`src.features.feature_extractor.extract_features` (paired-curve stats).

    Returns
    -------
    dict
        Output of :func:`src.models.training_pipeline.train_model`.
    """
    print("🔄 Training AI Model with 4-feature signature...")

    X = [
        [0.05, 0.02, 0.1, 0.998],
        [0.15, 0.08, 0.3, 0.985],
        [5.2, 3.1, 14.5, 0.65],
        [4.8, 2.5, 11.0, 0.71],
        [1.8, 1.2, 4.5, 0.88],
        [2.1, 1.4, 5.2, 0.85],
    ]

    y = [
        "Healthy",
        "Healthy",
        "Winding Deformation",
        "Winding Deformation",
        "Insulation Degradation",
        "Insulation Degradation",
    ]

    result = train_classifier(X, y, test_size=0.2, random_state=42, model_path=MODEL_PATH)
    print(f"✅ Model saved to {result['model_path']}")
    print(f"   Test accuracy: {result['test_accuracy']:.4f}")
    return result


def load_model():
    """
    Load the trained classifier from disk; train the demo model if missing or corrupt.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        The loaded model instance.
    """
    if not os.path.exists(MODEL_PATH):
        train_demo_model()

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        train_demo_model()
        return joblib.load(MODEL_PATH)

"""
Feature extraction and preprocessing for FRA curves.

- ``feature_extractor``: preprocessing and paired-curve ML features.
- ``fra_signal_features``: single-curve peaks, bands, smoothness, optional reference correlation.
"""

from src.features.fra_signal_features import (
    extract_fra_signal_features,
    feature_dict_for_ui,
    fra_features_to_vector,
    generate_feature_insights,
)

__all__ = [
    "extract_fra_signal_features",
    "feature_dict_for_ui",
    "fra_features_to_vector",
    "generate_feature_insights",
]

"""
Expert rules and UI-oriented guidance for FRA diagnostics.

- ``rules``: correlation bands and text recommendations.
- ``engine``: structured rule evaluation (peak shift, band energy).
"""

from src.expert.engine import evaluate_expert_rules

__all__ = ["evaluate_expert_rules"]

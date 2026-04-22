"""
Public package interface for the adaptive-LASSO estimator.

The package exposes an OLS-compatible model class, a convenience alias using
the repository's original capitalization, and a one-shot fitting helper for
scripts that do not need to keep the intermediate model object around.
"""

from .model import AdaptiveLasso, AdaptiveLassoResults, AdaptiveLassoResultsWrapper, AdaptiveLASSO, fit_adaptive_lasso

__all__ = [
    "AdaptiveLasso",
    "AdaptiveLassoResults",
    "AdaptiveLassoResultsWrapper",
    "AdaptiveLASSO",
    "fit_adaptive_lasso",
]

__version__ = "1.0.0"

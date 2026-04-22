"""
Minimal adaptive-LASSO example.

This script creates a synthetic linear-regression problem with a sparse true
coefficient vector, fits the adaptive-LASSO estimator, and prints the selected
variables. It is intentionally small so it can double as both a smoke test and
an executable usage example for the README.
"""

import numpy as np
import statsmodels.api as sm

from adaptive_lasso import AdaptiveLasso


def main() -> None:
    """Run a small end-to-end adaptive-LASSO example."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(250, 6))
    beta = np.array([2.0, -1.25, 0.0, 0.0, 0.8, 0.0])
    y = x @ beta + rng.normal(scale=0.4, size=250)

    # Leave the intercept unpenalized by using a leading zero in the penalty
    # vector, matching the convention described in the package README.
    exog = sm.add_constant(x)
    alpha = np.array([0.0, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

    model = AdaptiveLasso(y, exog)
    result = model.fit_regularized(alpha=alpha, adaptive_maxiter=15)

    print(result.params)
    print("Selected variables:", result.selected_variables)


if __name__ == "__main__":
    main()

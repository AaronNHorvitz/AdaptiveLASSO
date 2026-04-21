"""Minimal adaptive-LASSO example."""

import numpy as np
import statsmodels.api as sm

from adaptive_lasso import AdaptiveLasso


def main() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(250, 6))
    beta = np.array([2.0, -1.25, 0.0, 0.0, 0.8, 0.0])
    y = x @ beta + rng.normal(scale=0.4, size=250)

    exog = sm.add_constant(x)
    alpha = np.array([0.0, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

    model = AdaptiveLasso(y, exog)
    result = model.fit_regularized(alpha=alpha, adaptive_maxiter=15)

    print(result.params)
    print("Selected variables:", result.selected_variables)


if __name__ == "__main__":
    main()

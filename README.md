# adaptive-lasso

`adaptive-lasso` is a small Python package that adds an adaptive LASSO estimator with a Statsmodels-style interface for linear regression.

The public API is intentionally close to `statsmodels.regression.linear_model.OLS`:

```python
from adaptive_lasso import AdaptiveLasso

model = AdaptiveLasso(endog, exog)
result = model.fit_regularized(alpha=0.1, L1_wt=1.0)
```

If you already know how to call `statsmodels.OLS(...).fit_regularized(...)` for LASSO, this package is designed to feel familiar. The main difference is that the penalization is adaptive: each coefficient receives its own data-driven L1 weight.

## Why this exists

Statsmodels exposes regularized linear-model fitting through `OLS.fit_regularized`, including LASSO when `L1_wt=1.0`. This package keeps that workflow but adds an adaptive LASSO estimator for users who want coefficient-specific penalty weights without leaving the Statsmodels ecosystem.

The methodology follows Hui Zou's 2006 paper:

- Hui Zou, *The Adaptive Lasso and Its Oracle Properties*, Journal of the American Statistical Association, 101(476), 1418-1429, 2006.
- DOI: `10.1198/016214506000000735`
- Publisher page: https://www.tandfonline.com/doi/abs/10.1198/016214506000000735

## Installation

```bash
pip install -r requirements.txt
```

For development:

```bash
pip install -r requirements-dev.txt
pytest
```

## Quick start

```python
import numpy as np
import statsmodels.api as sm

from adaptive_lasso import AdaptiveLasso

rng = np.random.default_rng(42)
n = 200

X = rng.normal(size=(n, 6))
beta = np.array([2.0, -1.5, 0.0, 0.0, 0.75, 0.0])
y = X @ beta + rng.normal(scale=0.5, size=n)

X = sm.add_constant(X)
alpha = np.array([0.0, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

model = AdaptiveLasso(y, X)
result = model.fit_regularized(
    alpha=alpha,
    L1_wt=1.0,
    adaptive_maxiter=15,
    adaptive_tol=1e-6,
    weight_exponent=1.0,
)

print(result.params)
print(result.selected_variables)
```

You can also use formulas:

```python
import pandas as pd

from adaptive_lasso import AdaptiveLasso

df = pd.DataFrame(
    {
        "y": y,
        "x1": X[:, 1],
        "x2": X[:, 2],
        "x3": X[:, 3],
        "x4": X[:, 4],
        "x5": X[:, 5],
        "x6": X[:, 6],
    }
)

model = AdaptiveLasso.from_formula("y ~ x1 + x2 + x3 + x4 + x5 + x6", data=df)
result = model.fit_regularized(alpha=0.08, L1_wt=1.0)
```

## API notes

- The constructor matches `statsmodels.OLS(endog, exog=None, missing='none', hasconst=None, **kwargs)`.
- `fit_regularized` accepts the familiar Statsmodels arguments: `alpha`, `L1_wt`, `start_params`, `profile_scale`, and `refit`.
- The extra adaptive-LASSO controls are:
  - `adaptive_maxiter`: maximum number of outer reweighting iterations.
  - `adaptive_tol`: stopping tolerance for the outer loop.
  - `weight_exponent`: the adaptive penalty exponent, often written as `gamma`.
  - `weight_eps`: a small positive constant that prevents divide-by-zero in the adaptive weights.
- `L1_wt` must be `1.0`. This package implements adaptive LASSO, not adaptive elastic net.
- Like Statsmodels OLS, no intercept is added automatically unless you add one yourself or use a formula.
- If you want an unpenalized intercept, pass `alpha` as a vector with a leading zero for the constant term.

## What the result object includes

The returned regularized results object exposes familiar attributes such as:

- `params`
- `fittedvalues`
- `predict(...)`

It also adds adaptive-LASSO-specific metadata:

- `selected_mask`
- `selected_variables`
- `adaptive_weights`
- `effective_alpha`
- `fit_history`
- `converged`

If `refit=True`, the selected model is refit with ordinary least squares on the active set, following the same broad idea as Statsmodels' regularized refit flow.

## Development status

This repository currently targets linear regression through an `OLS`-compatible model class. It does not yet implement adaptive regularization for `WLS`, `GLS`, or `GLM`.

## Acknowledgements

An earlier version of this repository was an exploratory script that referenced Alexandre Gramfort's public adaptive-LASSO gist. The current codebase is a fresh rewrite built around Statsmodels' public OLS API and the adaptive-LASSO literature above.


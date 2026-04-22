# adaptive-lasso

`adaptive-lasso` is a small Python package that adds an adaptive LASSO estimator with a Statsmodels-style interface for linear regression.

This repository began as a small adaptive LASSO experiment in 2019, informed
by public examples and the academic literature. It was later revisited and
rewritten into a small public package with a Statsmodels-style API, tests,
clearer documentation, and a Jupyter notebook.

The adaptive LASSO method itself comes from Hui Zou's 2006 paper, and this
repository preserves attribution to Alexandre Gramfort's public GitHub demo
that influenced the earlier exploratory version. To be clear, this repository
does not claim authorship of the adaptive LASSO method or of the earlier demo
that inspired the original experiment. The current codebase is a modern
rewrite intended to make the method easier to use in a familiar Python
workflow.

An end-to-end walkthrough using a synthetic sparse dataset is available in
[notebooks/adaptive_lasso_dummy_data.ipynb](/var/home/aaronnhorvitz/dev/AdaptiveLASSO/notebooks/adaptive_lasso_dummy_data.ipynb).

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

The statistical method is academic, but the specific Statsmodels-style API in
this repository is a software design choice rather than a direct reproduction
of a single canonical academic implementation. In other words: the adaptive
LASSO idea comes from the literature, while the `statsmodels.OLS`-like
interface in this package is the practical packaging decision made for this
repository.

## What Adaptive LASSO Is

Adaptive LASSO is a sparse linear-regression method that starts from the same
general goal as ordinary LASSO: estimate a coefficient vector while shrinking
weak or noisy predictors toward zero. The key idea is that it does **not**
penalize every coefficient equally. Instead, it uses an initial "pilot"
estimate to decide which coefficients should be penalized more heavily and
which should be penalized less.

For a linear model with response vector $y \in \mathbb{R}^n$, design matrix
$X \in \mathbb{R}^{n \times p}$, and coefficient vector
$\beta \in \mathbb{R}^p$, ordinary least squares minimizes

$$
\frac{1}{2n}\lVert y - X\beta \rVert_2^2.
$$

Standard LASSO adds a uniform $\ell_1$ penalty:

$$
\hat{\beta}^{\mathrm{lasso}}
=
\arg\min_{\beta}
\left\{
\frac{1}{2n}\lVert y - X\beta \rVert_2^2
+
\lambda \sum_{j=1}^p |\beta_j|
\right\}.
$$

That penalty encourages sparsity because the $\ell_1$ term can drive some
coefficients exactly to zero. The downside is that the same penalty strength is
applied to every coefficient, which can introduce extra bias for large true
signals.

Adaptive LASSO modifies the penalty so that each coefficient gets its own
weight:

$$
\hat{\beta}^{\mathrm{adapt}}
=
\arg\min_{\beta}
\left\{
\frac{1}{2n}\lVert y - X\beta \rVert_2^2
+
\lambda \sum_{j=1}^p w_j |\beta_j|
\right\},
$$

where the adaptive weights are typically defined using a pilot estimate
$\tilde{\beta}$:

$$
w_j = \frac{1}{|\tilde{\beta}_j|^{\gamma} + \varepsilon},
\qquad \gamma > 0.
$$

Here:

- $\tilde{\beta}_j$ is the pilot estimate for coefficient $j$.
- $\gamma$ controls how aggressively the weighting distinguishes strong and
  weak predictors.
- $\varepsilon$ is a small positive constant used in practice to avoid
  division by zero when a pilot coefficient is exactly zero.

This weighting scheme gives the method its name:

- If $|\tilde{\beta}_j|$ is large, then $w_j$ is small, so coefficient $j$
  receives a lighter penalty.
- If $|\tilde{\beta}_j|$ is close to zero, then $w_j$ is large, so coefficient
  $j$ is penalized more strongly.

Intuitively, adaptive LASSO says: "if the pilot fit already suggests this
predictor matters, be gentler with it in the sparse fit; if the pilot fit says
it is weak, shrink it harder."

Under suitable regularity conditions, this reweighting can improve variable
selection relative to plain LASSO and can recover the so-called *oracle
property*: asymptotically, the method can identify the correct active set and
estimate the nonzero coefficients as if the irrelevant variables had been known
in advance.

## How This Package Implements It

This package uses a Statsmodels-style outer reweighting loop around the OLS
regularized solver:

1. Fit or accept a pilot estimate $\tilde{\beta}$.
2. Compute adaptive weights
   $w_j = 1 / (|\tilde{\beta}_j|^{\gamma} + \varepsilon)$.
3. Solve a weighted LASSO problem.
4. Update the weights from the new coefficient vector.
5. Repeat until the coefficients stop changing by more than the specified
   tolerance or the maximum number of outer iterations is reached.

In the package API, the user-facing `alpha` corresponds to the base penalty
level, and the actual coefficient-specific penalty applied by the solver is

$$
\alpha_j^{\mathrm{effective}} = \alpha_j \, w_j.
$$

That design lets you do familiar Statsmodels-style things such as leaving the
intercept unpenalized by passing a leading zero, for example
`alpha=[0.0, 0.1, 0.1, ...]`.

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

## Provenance And Acknowledgements

The adaptive LASSO algorithm itself comes from the statistical literature,
especially:

- Hui Zou, *The Adaptive Lasso and Its Oracle Properties*, Journal of the
  American Statistical Association, 101(476), 1418-1429, 2006.

This repository also has a more specific implementation history:

- An earlier version of this repository began as a small personal exploratory
  script written in 2019.
- That exploratory version was informed by Alexandre Gramfort's public
  adaptive-LASSO gist:
  https://gist.github.com/agramfort/1610922
- Gramfort's gist is published under the BSD 3-clause license.
- The current repository is a substantial rewrite completed in 2026, with a
  Statsmodels-style API, packaging, tests, documentation, and notebook
  examples.

In other words: the algorithm is academic, the earliest repository version was
influenced by a public demo, and the current package is a modern rewrite rather
than a direct copy of that original exploratory script.

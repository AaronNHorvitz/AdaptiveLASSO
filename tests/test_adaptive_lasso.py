import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.base.elastic_net import RegularizedResultsWrapper
from statsmodels.regression.linear_model import RegressionResults

from adaptive_lasso import AdaptiveLasso, AdaptiveLASSO, fit_adaptive_lasso


def make_regression_data(seed: int = 42, n: int = 300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    beta = np.array([2.5, -1.5, 0.0, 0.0, 1.0, 0.0])
    y = x @ beta + rng.normal(scale=0.5, size=n)
    return x, y, beta


def make_named_exog(x: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(1, x.shape[1] + 1)])
    return sm.add_constant(df, has_constant="add")


def test_constructor_matches_statsmodels_ols_shape():
    x, y, _ = make_regression_data()
    exog = sm.add_constant(x)

    model = AdaptiveLasso(y, exog)

    assert model.endog.shape == y.shape
    assert model.exog.shape == exog.shape


def test_fit_regularized_returns_statsmodels_style_wrapper():
    x, y, _ = make_regression_data()
    exog = make_named_exog(x)
    alpha = np.array([0.0] + [0.08] * x.shape[1])

    result = AdaptiveLasso(y, exog).fit_regularized(alpha=alpha, L1_wt=1.0)

    assert isinstance(result, RegularizedResultsWrapper)
    assert result.params.shape == (exog.shape[1],)
    assert result.converged is True
    assert "const" in result.params.index
    assert isinstance(result.selected_variables, list)


def test_formula_interface_is_supported():
    x, y, _ = make_regression_data()
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(1, 7)])
    df["y"] = y

    model = AdaptiveLasso.from_formula("y ~ x1 + x2 + x3 + x4 + x5 + x6", data=df)
    result = model.fit_regularized(alpha=0.08, L1_wt=1.0)

    assert "Intercept" in result.params.index
    assert len(result.selected_variables) >= 2


def test_refit_returns_regression_results():
    x, y, _ = make_regression_data()
    exog = make_named_exog(x)
    alpha = np.array([0.0] + [0.08] * x.shape[1])

    result = AdaptiveLasso(y, exog).fit_regularized(alpha=alpha, refit=True)

    assert isinstance(result, RegressionResults)
    assert result.regularized is True
    assert result.method == "adaptive_lasso"
    assert result.params.shape == (exog.shape[1],)


def test_noise_features_are_zeroed_out_under_reasonable_penalty():
    x, y, _ = make_regression_data()
    exog = make_named_exog(x)
    alpha = np.array([0.0] + [0.10] * x.shape[1])

    result = AdaptiveLasso(y, exog).fit_regularized(
        alpha=alpha,
        adaptive_maxiter=15,
        adaptive_tol=1e-7,
    )

    zero_like = result.params.loc[["x3", "x4", "x6"]]
    assert np.all(np.abs(zero_like.to_numpy()) < 1e-4)


def test_l1_weight_other_than_one_is_rejected():
    x, y, _ = make_regression_data()
    exog = sm.add_constant(x)

    with pytest.raises(ValueError, match="L1_wt=1.0"):
        AdaptiveLasso(y, exog).fit_regularized(alpha=0.1, L1_wt=0.5)


def test_unknown_method_is_rejected():
    x, y, _ = make_regression_data()
    exog = sm.add_constant(x)

    with pytest.raises(ValueError, match="method must be one of"):
        AdaptiveLasso(y, exog).fit_regularized(method="elastic_net", alpha=0.1)


def test_convenience_function_and_alias_work():
    x, y, _ = make_regression_data()
    exog = make_named_exog(x)

    result_from_function = fit_adaptive_lasso(y, exog, alpha=0.08)
    result_from_alias = AdaptiveLASSO(y, exog).fit_regularized(alpha=0.08)

    assert result_from_function.params.shape == result_from_alias.params.shape
    assert np.allclose(np.asarray(result_from_function.params), np.asarray(result_from_alias.params))


def test_alpha_vector_can_leave_intercept_unpenalized():
    x, y, _ = make_regression_data()
    exog = make_named_exog(x)
    alpha = np.array([0.0] + [0.08] * x.shape[1])

    result = AdaptiveLasso(y, exog).fit_regularized(alpha=alpha)

    assert result.effective_alpha.iloc[0] == pytest.approx(0.0)
    assert result.params.index[0] == "const"

"""Adaptive LASSO model built on top of statsmodels OLS."""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.base.elastic_net import RegularizedResults, RegularizedResultsWrapper as StatsmodelsRegularizedResultsWrapper
import statsmodels.base.wrapper as wrap
from statsmodels.regression.linear_model import OLS

_METHOD_ALIASES = {"adaptive_lasso", "adaptive-lasso", "alasso"}
_SUPPORTED_KWARGS = {
    "maxiter",
    "cnvrg_tol",
    "zero_tol",
    "adaptive_maxiter",
    "adaptive_tol",
    "weight_exponent",
    "weight_eps",
}


def _coerce_alpha(alpha: float | np.ndarray, n_params: int) -> np.ndarray:
    """Return a coefficient-wise penalty vector."""
    if np.isscalar(alpha):
        return np.full(n_params, float(alpha), dtype=float)

    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.shape != (n_params,):
        raise ValueError(
            f"alpha must be a scalar or have shape ({n_params},), got {alpha_array.shape}."
        )
    return alpha_array


def _coerce_start_params(start_params: np.ndarray | None, n_params: int) -> np.ndarray | None:
    """Normalize an optional coefficient vector."""
    if start_params is None:
        return None

    params = np.asarray(start_params, dtype=float)
    if params.shape != (n_params,):
        raise ValueError(
            f"start_params must have shape ({n_params},), got {params.shape}."
        )
    return params


def _adaptive_weights(params: np.ndarray, weight_exponent: float, weight_eps: float) -> np.ndarray:
    """Compute adaptive penalty weights."""
    return 1.0 / (np.abs(params) ** weight_exponent + weight_eps)


def _selected_variables(model: OLS, selected_mask: np.ndarray) -> list[str]:
    """Return selected variable names when they are available."""
    exog_names = getattr(model, "exog_names", None)
    if exog_names is None:
        return [str(index) for index in np.flatnonzero(selected_mask)]
    return [name for name, keep in zip(exog_names, selected_mask) if keep]


class AdaptiveLassoResults(RegularizedResults):
    """Regularized results with adaptive-LASSO-specific metadata."""

    def __init__(
        self,
        model: OLS,
        params: np.ndarray,
        *,
        base_alpha: float | np.ndarray,
        effective_alpha: np.ndarray,
        initial_params: np.ndarray,
        adaptive_weights: np.ndarray,
        fit_history: dict[str, Any],
        converged: bool,
        zero_tol: float,
    ) -> None:
        super().__init__(model, params)
        self.base_alpha = base_alpha
        self.effective_alpha = np.asarray(effective_alpha, dtype=float)
        self.initial_params = np.asarray(initial_params, dtype=float)
        self.adaptive_weights = np.asarray(adaptive_weights, dtype=float)
        self.fit_history = fit_history
        self.converged = converged
        self.zero_tol = float(zero_tol)
        self.method = "adaptive_lasso"
        self.regularized = True
        self.selected_mask = np.abs(np.asarray(params, dtype=float)) > self.zero_tol
        self.selected_variables = _selected_variables(model, self.selected_mask)


class AdaptiveLassoResultsWrapper(StatsmodelsRegularizedResultsWrapper):
    """Statsmodels-style wrapper for adaptive-LASSO results."""

    _attrs = {
        "params": "columns",
        "resid": "rows",
        "fittedvalues": "rows",
        "selected_mask": "columns",
        "initial_params": "columns",
        "adaptive_weights": "columns",
        "effective_alpha": "columns",
    }
    _wrap_attrs = _attrs
    _wrap_methods = {"predict": "rows"}


wrap.populate_wrapper(AdaptiveLassoResultsWrapper, AdaptiveLassoResults)


class AdaptiveLasso(OLS):
    """OLS-compatible model with an adaptive-LASSO `fit_regularized` method."""

    def fit_regularized(
        self,
        method: str = "adaptive_lasso",
        alpha: float | np.ndarray = 0.0,
        L1_wt: float = 1.0,
        start_params: np.ndarray | None = None,
        profile_scale: bool = False,
        refit: bool = False,
        **kwargs: Any,
    ):
        """
        Fit an adaptive LASSO model.

        Parameters largely match ``statsmodels.OLS.fit_regularized``.

        Notes
        -----
        ``start_params`` serves two roles in this implementation:

        - it seeds the first weighted LASSO solve, and
        - it defines the first set of adaptive weights.

        If ``start_params`` is omitted, an unpenalized OLS fit is used as the
        pilot estimate.
        """

        if method not in _METHOD_ALIASES:
            raise ValueError(
                "method must be one of "
                f"{sorted(_METHOD_ALIASES)}, got {method!r}."
            )

        if L1_wt != 1.0:
            raise ValueError(
                "AdaptiveLasso currently supports only pure L1 penalization; "
                "please pass L1_wt=1.0."
            )

        unknown_kwargs = sorted(set(kwargs) - _SUPPORTED_KWARGS)
        if unknown_kwargs:
            unknown = ", ".join(unknown_kwargs)
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        options = {
            "maxiter": 50,
            "cnvrg_tol": 1e-10,
            "zero_tol": 1e-8,
            "adaptive_maxiter": 10,
            "adaptive_tol": 1e-6,
            "weight_exponent": 1.0,
            "weight_eps": 1e-6,
        }
        options.update(kwargs)

        zero_tol = float(options["zero_tol"])
        adaptive_tol = float(options["adaptive_tol"])
        adaptive_maxiter = int(options["adaptive_maxiter"])
        weight_exponent = float(options["weight_exponent"])
        weight_eps = float(options["weight_eps"])
        solver_maxiter = int(options["maxiter"])
        solver_cnvrg_tol = float(options["cnvrg_tol"])

        if adaptive_maxiter < 1:
            raise ValueError("adaptive_maxiter must be at least 1.")
        if weight_eps <= 0:
            raise ValueError("weight_eps must be strictly positive.")
        if weight_exponent < 0:
            raise ValueError("weight_exponent must be non-negative.")

        n_params = int(self.exog.shape[1])
        alpha_array = _coerce_alpha(alpha, n_params)
        pilot_params = _coerce_start_params(start_params, n_params)
        if pilot_params is None:
            pilot_params = np.asarray(self.fit().params, dtype=float)

        params = pilot_params.copy()
        fit_history: dict[str, Any] = {
            "objective": [],
            "param_change": [],
            "solver_maxiter": solver_maxiter,
            "solver_cnvrg_tol": solver_cnvrg_tol,
            "adaptive_maxiter": adaptive_maxiter,
            "adaptive_tol": adaptive_tol,
            "weight_exponent": weight_exponent,
            "weight_eps": weight_eps,
            "zero_tol": zero_tol,
        }

        converged = False
        effective_alpha = alpha_array * _adaptive_weights(
            params,
            weight_exponent=weight_exponent,
            weight_eps=weight_eps,
        )

        for _ in range(adaptive_maxiter):
            result = super().fit_regularized(
                method="elastic_net",
                alpha=effective_alpha,
                L1_wt=1.0,
                start_params=params,
                profile_scale=profile_scale,
                refit=False,
                maxiter=solver_maxiter,
                cnvrg_tol=solver_cnvrg_tol,
                zero_tol=zero_tol,
            )

            new_params = np.array(result.params, dtype=float, copy=True)
            new_params[np.abs(new_params) < zero_tol] = 0.0

            change = float(np.max(np.abs(new_params - params)))
            fit_history["param_change"].append(change)
            fit_history["objective"].append(self._objective(new_params, effective_alpha))

            params = new_params
            effective_alpha = alpha_array * _adaptive_weights(
                params,
                weight_exponent=weight_exponent,
                weight_eps=weight_eps,
            )

            if change < adaptive_tol:
                converged = True
                break

        fit_history["iterations"] = len(fit_history["param_change"])

        if refit:
            return self._refit_selected(
                params=params,
                zero_tol=zero_tol,
                converged=converged,
                fit_history=fit_history,
                base_alpha=alpha,
                effective_alpha=effective_alpha,
                initial_params=pilot_params,
                adaptive_weights=_adaptive_weights(
                    params,
                    weight_exponent=weight_exponent,
                    weight_eps=weight_eps,
                ),
            )

        results = AdaptiveLassoResults(
            self,
            params,
            base_alpha=alpha,
            effective_alpha=effective_alpha,
            initial_params=pilot_params,
            adaptive_weights=_adaptive_weights(
                params,
                weight_exponent=weight_exponent,
                weight_eps=weight_eps,
            ),
            fit_history=fit_history,
            converged=converged,
            zero_tol=zero_tol,
        )
        return AdaptiveLassoResultsWrapper(results)

    def _objective(self, params: np.ndarray, effective_alpha: np.ndarray) -> float:
        """Penalized least-squares objective used for fit history."""
        resid = np.asarray(self.endog, dtype=float) - np.asarray(self.exog, dtype=float) @ params
        rss = np.dot(resid, resid)
        penalty = np.dot(effective_alpha, np.abs(params))
        return float(0.5 * rss / float(self.nobs) + penalty)

    def _refit_selected(
        self,
        *,
        params: np.ndarray,
        zero_tol: float,
        converged: bool,
        fit_history: dict[str, Any],
        base_alpha: float | np.ndarray,
        effective_alpha: np.ndarray,
        initial_params: np.ndarray,
        adaptive_weights: np.ndarray,
    ):
        """Refit the active set with ordinary least squares."""
        selected = np.flatnonzero(np.abs(params) > zero_tol)
        k_exog = self.exog.shape[1]
        cov = np.zeros((k_exog, k_exog))
        init_args = {key: getattr(self, key, None) for key in self._init_keys}
        params = np.asarray(params, dtype=float).copy()

        if len(selected) > 0:
            model_refit = self.__class__(self.endog, self.exog[:, selected], **init_args)
            fitted = model_refit.fit()
            params[selected] = np.asarray(fitted.params, dtype=float)
            cov[np.ix_(selected, selected)] = np.asarray(fitted.normalized_cov_params, dtype=float)
        else:
            model_refit = self.__class__(self.endog, self.exog[:, 0], **init_args)
            fitted = model_refit.fit(maxiter=0)

        if issubclass(fitted.__class__, wrap.ResultsWrapper):
            klass = fitted._results.__class__
        else:
            klass = fitted.__class__

        scale = getattr(fitted, "scale", 1.0)
        df_model, df_resid = self.df_model, self.df_resid
        self.df_model = len(selected)
        self.df_resid = self.nobs - self.df_model
        refit_results = klass(self, params, cov, scale=scale)
        self.df_model, self.df_resid = df_model, df_resid

        refit_results.base_alpha = base_alpha
        refit_results.effective_alpha = np.asarray(effective_alpha, dtype=float)
        refit_results.initial_params = np.asarray(initial_params, dtype=float)
        refit_results.adaptive_weights = np.asarray(adaptive_weights, dtype=float)
        refit_results.fit_history = fit_history
        refit_results.converged = converged
        refit_results.zero_tol = float(zero_tol)
        refit_results.method = "adaptive_lasso"
        refit_results.regularized = True
        refit_results.selected_mask = np.abs(params) > zero_tol
        refit_results.selected_variables = _selected_variables(self, refit_results.selected_mask)
        return refit_results


AdaptiveLASSO = AdaptiveLasso


def fit_adaptive_lasso(endog, exog=None, **kwargs):
    """Convenience function for one-shot fitting."""
    model = AdaptiveLasso(endog=endog, exog=exog)
    return model.fit_regularized(**kwargs)

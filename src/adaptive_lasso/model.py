"""
Adaptive LASSO model built on top of :class:`statsmodels.regression.linear_model.OLS`.

The implementation in this module intentionally mirrors the public ergonomics of
``statsmodels.OLS.fit_regularized`` while replacing the uniform L1 penalty with
the adaptive penalty proposed by Zou (2006). The outer loop repeatedly solves a
weighted LASSO problem, updating the coefficient-specific penalty weights after
each solve.
"""

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
    """
    Normalize a scalar or vector penalty specification.

    Parameters
    ----------
    alpha : float or ndarray
        Penalty level passed to :meth:`AdaptiveLasso.fit_regularized`. Scalars
        are broadcast across every coefficient. One-dimensional arrays are
        interpreted as coefficient-specific base penalties.
    n_params : int
        Number of model coefficients, including any intercept term present in
        ``exog``.

    Returns
    -------
    ndarray of shape (n_params,)
        A floating-point vector containing one base penalty per coefficient.

    Raises
    ------
    ValueError
        If ``alpha`` is an array with a shape different from ``(n_params,)``.

    Notes
    -----
    Keeping the base penalty separate from the adaptive weights makes it easy
    to mimic the common Statsmodels pattern of leaving an intercept
    unpenalized, e.g. ``alpha=[0.0, 0.1, 0.1, ...]``.
    """
    if np.isscalar(alpha):
        return np.full(n_params, float(alpha), dtype=float)

    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.shape != (n_params,):
        raise ValueError(
            f"alpha must be a scalar or have shape ({n_params},), got {alpha_array.shape}."
        )
    return alpha_array


def _coerce_start_params(start_params: np.ndarray | None, n_params: int) -> np.ndarray | None:
    """
    Validate an optional starting-value vector.

    Parameters
    ----------
    start_params : ndarray or None
        Optional initial coefficient vector. When provided, these values serve
        as both the warm start for the first weighted LASSO solve and the pilot
        estimate used to construct the first adaptive weights.
    n_params : int
        Number of model coefficients expected by the design matrix.

    Returns
    -------
    ndarray or None
        A floating-point copy-ready array if input is supplied, otherwise
        ``None``.

    Raises
    ------
    ValueError
        If ``start_params`` does not have shape ``(n_params,)``.
    """
    if start_params is None:
        return None

    params = np.asarray(start_params, dtype=float)
    if params.shape != (n_params,):
        raise ValueError(
            f"start_params must have shape ({n_params},), got {params.shape}."
        )
    return params


def _adaptive_weights(params: np.ndarray, weight_exponent: float, weight_eps: float) -> np.ndarray:
    """
    Compute coefficient-specific adaptive penalty weights.

    Parameters
    ----------
    params : ndarray
        Pilot or current coefficient estimates.
    weight_exponent : float
        Adaptive exponent, often denoted by ``gamma`` in the literature.
    weight_eps : float
        Positive numerical stabilizer added to the denominator to prevent
        infinite weights when a coefficient is exactly zero.

    Returns
    -------
    ndarray
        Weight vector defined by
        ``1 / (abs(params) ** weight_exponent + weight_eps)``.

    Notes
    -----
    Large pilot coefficients receive smaller penalties in the next weighted
    LASSO solve, while coefficients near zero receive larger penalties.
    """
    return 1.0 / (np.abs(params) ** weight_exponent + weight_eps)


def _selected_variables(model: OLS, selected_mask: np.ndarray) -> list[str]:
    """
    Convert a boolean support mask into human-readable variable names.

    Parameters
    ----------
    model : OLS
        Fitted or initialized OLS-compatible model carrying ``exog_names``.
    selected_mask : ndarray of bool
        Boolean mask indicating which coefficients are considered active.

    Returns
    -------
    list of str
        Selected variable names when the model exposes names, otherwise string
        versions of the selected positional indices.
    """
    exog_names = getattr(model, "exog_names", None)
    if exog_names is None:
        return [str(index) for index in np.flatnonzero(selected_mask)]
    return [name for name, keep in zip(exog_names, selected_mask) if keep]


class AdaptiveLassoResults(RegularizedResults):
    """
    Results object for adaptive-LASSO fits without post-selection refitting.

    Parameters
    ----------
    model : OLS
        The model instance used for estimation.
    params : ndarray
        Estimated coefficient vector on the original design scale.
    base_alpha : float or ndarray
        User-supplied base penalty before adaptive reweighting.
    effective_alpha : ndarray
        Final coefficient-specific penalty vector after combining
        ``base_alpha`` with the adaptive weights from the last outer
        iteration.
    initial_params : ndarray
        Pilot estimate used to initialize the adaptive weighting scheme.
    adaptive_weights : ndarray
        Final adaptive weights associated with ``effective_alpha``.
    fit_history : dict
        Iteration diagnostics collected during the outer reweighting loop.
    converged : bool
        Indicator for whether the outer loop met ``adaptive_tol`` before
        exhausting ``adaptive_maxiter``.
    zero_tol : float
        Coefficients with absolute value below this threshold are treated as
        exact zeros when deriving support information.

    Attributes
    ----------
    selected_mask : ndarray of bool
        Boolean indicator of the selected support.
    selected_variables : list of str
        Human-readable names of the selected variables when available.

    Notes
    -----
    This class extends Statsmodels' ``RegularizedResults`` with a few pieces of
    metadata that are particularly useful for sparse-model workflows, such as
    the final support and the adaptive penalty weights.
    """

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
    """
    Statsmodels-style wrapper for :class:`AdaptiveLassoResults`.

    Notes
    -----
    The wrapper preserves pandas row and column labels for common outputs such
    as ``params`` and ``fittedvalues`` when the underlying Statsmodels data
    container has that metadata available.
    """

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
    """
    Ordinary least squares model with adaptive-LASSO regularization.

    Parameters
    ----------
    endog : array_like
        Response variable. This matches the ``statsmodels.OLS`` constructor.
    exog : array_like, optional
        Design matrix. No intercept is added automatically unless one is
        already present in ``exog`` or the model is created with
        :meth:`from_formula`.
    missing : {"none", "drop", "raise"}, default "none"
        Missing-data handling strategy inherited from Statsmodels.
    hasconst : bool or None, default None
        Whether ``exog`` already contains a constant column.
    **kwargs
        Additional keyword arguments passed through to the base
        :class:`statsmodels.regression.linear_model.OLS` initializer.

    Notes
    -----
    This class does not re-implement OLS mechanics. Instead, it subclasses
    Statsmodels' OLS class and reuses its existing regularized solver inside an
    outer adaptive-weight update loop.

    See Also
    --------
    statsmodels.regression.linear_model.OLS
        Base model class that supplies the constructor, formula interface, and
        standard least-squares machinery.
    """

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

        Parameters
        ----------
        method : {"adaptive_lasso", "adaptive-lasso", "alasso"}, \
                default "adaptive_lasso"
            Name of the regularization method. Aliases are accepted for
            convenience, but only adaptive LASSO is implemented.
        alpha : float or ndarray, default 0.0
            Base penalty weight. A scalar applies the same baseline penalty to
            every coefficient. A vector of length ``k_params`` allows
            coefficient-specific penalties. Passing a leading zero is the
            standard way to leave an intercept unpenalized.
        L1_wt : float, default 1.0
            L1 penalty fraction, kept for API familiarity with Statsmodels.
            Adaptive LASSO corresponds to pure L1 regularization, so this must
            be ``1.0``.
        start_params : ndarray or None, default None
            Optional pilot estimate for the coefficients. When omitted, the
            method fits an unpenalized OLS model and uses those coefficients to
            seed the adaptive weights.
        profile_scale : bool, default False
            Passed through to Statsmodels' inner regularized solver for API
            compatibility. For OLS this primarily keeps behavior aligned with
            ``OLS.fit_regularized``.
        refit : bool, default False
            If ``True``, refit an unpenalized OLS model on the support selected
            by the adaptive-LASSO solution and return a
            ``RegressionResults``-style object. If ``False``, return a
            regularized results wrapper.
        **kwargs
            Additional configuration for the inner solver and outer adaptive
            loop.

            Supported keys are:

            ``maxiter`` : int, default 50
                Maximum number of coordinate-descent sweeps used by each inner
                weighted LASSO solve.
            ``cnvrg_tol`` : float, default 1e-10
                Convergence tolerance for the inner Statsmodels solver.
            ``zero_tol`` : float, default 1e-8
                Threshold below which coefficients are hard-thresholded to
                zero after each inner solve.
            ``adaptive_maxiter`` : int, default 10
                Maximum number of outer adaptive reweighting iterations.
            ``adaptive_tol`` : float, default 1e-6
                Outer-loop convergence tolerance based on the sup-norm change
                in successive parameter vectors.
            ``weight_exponent`` : float, default 1.0
                Adaptive weighting exponent, usually denoted ``gamma``.
            ``weight_eps`` : float, default 1e-6
                Positive stabilizer used in the denominator of the adaptive
                weight formula.

        Returns
        -------
        AdaptiveLassoResultsWrapper or RegressionResults
            Regularized adaptive-LASSO results when ``refit=False``. When
            ``refit=True``, an OLS results object is returned with additional
            adaptive-LASSO metadata attached.

        Raises
        ------
        ValueError
            If an unsupported method name is supplied, if ``L1_wt`` differs
            from ``1.0``, or if adaptive weighting options are invalid.
        TypeError
            If unsupported keyword arguments are passed in ``kwargs``.

        Notes
        -----
        The adaptive-LASSO algorithm is implemented as an outer loop around
        Statsmodels' existing L1-regularized OLS solver:

        1. Obtain a pilot estimate.
        2. Convert that estimate into coefficient-specific weights.
        3. Solve a weighted LASSO problem.
        4. Recompute the weights from the new coefficients.
        5. Repeat until the parameter change falls below ``adaptive_tol`` or
           ``adaptive_maxiter`` is reached.

        The optimization problem solved in each outer iteration is equivalent
        to a standard LASSO problem with penalty vector
        ``alpha * adaptive_weights``.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> from adaptive_lasso import AdaptiveLasso
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=(100, 3))
        >>> y = 1.5 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(size=100)
        >>> exog = sm.add_constant(x)
        >>> model = AdaptiveLasso(y, exog)
        >>> result = model.fit_regularized(alpha=[0.0, 0.1, 0.1, 0.1])
        >>> result.params.shape
        (4,)
        """
        # Validate the public API surface first so callers get immediate,
        # Statsmodels-like feedback when an unsupported option is supplied.
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

        # Split the mixed keyword bag into explicit outer-loop and inner-solver
        # settings to keep the algorithm easier to inspect and debug.
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
            # An unpenalized OLS estimate is the default pilot estimator in the
            # adaptive-LASSO literature and provides the first set of weights.
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
            # Each outer iteration is a plain LASSO solve with a coefficient-
            # specific penalty vector derived from the previous iterate.
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

            # Track the sup-norm change so we can report a simple, robust
            # convergence diagnostic for the outer adaptive loop.
            change = float(np.max(np.abs(new_params - params)))
            fit_history["param_change"].append(change)
            fit_history["objective"].append(self._objective(new_params, effective_alpha))

            params = new_params
            # Reweight the next LASSO problem using the latest coefficient
            # magnitudes; larger coefficients are penalized less heavily.
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
            # Post-selection refitting is often useful for interpretation and
            # for obtaining familiar OLS diagnostics on the active set.
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
        """
        Evaluate the weighted LASSO objective for the current iterate.

        Parameters
        ----------
        params : ndarray
            Coefficient vector to evaluate.
        effective_alpha : ndarray
            Coefficient-specific L1 penalty used in the current outer
            iteration.

        Returns
        -------
        float
            Penalized least-squares objective
            ``0.5 * RSS / nobs + sum(effective_alpha * abs(params))``.

        Notes
        -----
        This diagnostic is recorded for transparency in ``fit_history``. It is
        not used as the stopping criterion for the outer loop.
        """
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
        """
        Refit the active set with ordinary least squares.

        Parameters
        ----------
        params : ndarray
            Adaptive-LASSO coefficient vector used to determine the active set.
        zero_tol : float
            Support threshold used to decide which coefficients are non-zero.
        converged : bool
            Whether the outer adaptive loop converged before reaching its
            iteration limit.
        fit_history : dict
            Iteration diagnostics collected during adaptive fitting.
        base_alpha : float or ndarray
            User-supplied base penalty before adaptive reweighting.
        effective_alpha : ndarray
            Final coefficient-specific penalty vector.
        initial_params : ndarray
            Pilot coefficients used to initialize the adaptive fit.
        adaptive_weights : ndarray
            Final adaptive weights associated with ``effective_alpha``.

        Returns
        -------
        RegressionResults
            Statsmodels regression results for the selected active set, padded
            back out to the full coefficient dimension and annotated with
            adaptive-LASSO metadata.

        Notes
        -----
        This mirrors the broad behavior of Statsmodels' ``refit=True`` path for
        regularized models: estimate sparsity with the penalized fit, then run
        an ordinary least squares regression using only the selected columns.
        """
        selected = np.flatnonzero(np.abs(params) > zero_tol)
        k_exog = self.exog.shape[1]
        cov = np.zeros((k_exog, k_exog))
        init_args = {key: getattr(self, key, None) for key in self._init_keys}
        params = np.asarray(params, dtype=float).copy()

        if len(selected) > 0:
            # Refit only the active columns, then place those estimates back
            # into a full-length parameter vector for a familiar user-facing
            # shape.
            model_refit = self.__class__(self.endog, self.exog[:, selected], **init_args)
            fitted = model_refit.fit()
            params[selected] = np.asarray(fitted.params, dtype=float)
            cov[np.ix_(selected, selected)] = np.asarray(fitted.normalized_cov_params, dtype=float)
        else:
            # Statsmodels' regularized refit path also needs a fallback branch
            # when no variables are selected so that it can still construct a
            # valid results object.
            model_refit = self.__class__(self.endog, self.exog[:, 0], **init_args)
            fitted = model_refit.fit(maxiter=0)

        if issubclass(fitted.__class__, wrap.ResultsWrapper):
            klass = fitted._results.__class__
        else:
            klass = fitted.__class__

        scale = getattr(fitted, "scale", 1.0)
        df_model, df_resid = self.df_model, self.df_resid
        # Temporarily update the model degrees of freedom so the refit results
        # reflect the selected model rather than the original full design.
        self.df_model = len(selected)
        self.df_resid = self.nobs - self.df_model
        refit_results = klass(self, params, cov, scale=scale)
        self.df_model, self.df_resid = df_model, df_resid

        # Reattach adaptive-LASSO metadata so downstream users can inspect the
        # selection event even after refitting with plain OLS.
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
    """
    Fit an adaptive-LASSO model in a single call.

    Parameters
    ----------
    endog : array_like
        Response variable passed to :class:`AdaptiveLasso`.
    exog : array_like, optional
        Design matrix passed to :class:`AdaptiveLasso`.
    **kwargs
        Keyword arguments forwarded directly to
        :meth:`AdaptiveLasso.fit_regularized`.

    Returns
    -------
    AdaptiveLassoResultsWrapper or RegressionResults
        Result of :meth:`AdaptiveLasso.fit_regularized`.

    See Also
    --------
    AdaptiveLasso.fit_regularized
        Full adaptive-LASSO fitting interface.
    """
    model = AdaptiveLasso(endog=endog, exog=exog)
    return model.fit_regularized(**kwargs)

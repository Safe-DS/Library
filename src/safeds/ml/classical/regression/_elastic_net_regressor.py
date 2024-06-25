from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from warnings import warn

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound

from ._regressor import Regressor
from ...hyperparameters import Choice

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class ElasticNetRegressor(Regressor):
    """Elastic net regression.

    Parameters
    ----------
    alpha:
        Controls the regularization of the model. The higher the value, the more regularized it becomes.
        If 0, a linear model is used.
    lasso_ratio:
        Number between 0 and 1 that controls the ratio between Lasso and Ridge regularization. If 0, only Ridge
        regularization is used. If 1, only Lasso regularization is used.

    Raises
    ------
    OutOfBoundsError
        If `alpha` is negative or `lasso_ratio` is not between 0 and 1.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, *, alpha: float | Choice[float] = 1.0, lasso_ratio: float | Choice[float] = 0.5) -> None:
        super().__init__()

        # Validation
        if isinstance(alpha, Choice):
            for a in alpha:
                _check_bounds("alpha", a, lower_bound=_ClosedBound(0))
        else:
            _check_bounds("alpha", alpha, lower_bound=_ClosedBound(0))

        if isinstance(lasso_ratio, Choice):
            for lr in lasso_ratio:
                _check_bounds("lasso_ratio", lr, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(1))
        else:
            _check_bounds("lasso_ratio", lasso_ratio, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(1))

        # Hyperparameters
        self._alpha = alpha
        self._lasso_ratio = lasso_ratio

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._alpha,
            self._lasso_ratio,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def alpha(self) -> float | Choice[float]:
        """The regularization of the model."""
        return self._alpha

    @property
    def lasso_ratio(self) -> float | Choice[float]:
        """Rhe ratio between Lasso and Ridge regularization."""
        return self._lasso_ratio

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> ElasticNetRegressor:
        return ElasticNetRegressor(
            alpha=self._alpha,
            lasso_ratio=self._lasso_ratio,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.linear_model import ElasticNet as SklearnElasticNet
        from sklearn.linear_model import LinearRegression as sk_LinearRegression
        from sklearn.linear_model import Ridge as SklearnRidge
        from sklearn.linear_model import Lasso as SklearnLasso


        #TODO Does Linear Regression have priority over other models? Should this always be a linear model if alpha is zero or does the lasso ratio still mater in that case? Might have do modify the order of model creation here.
        if self._alpha == 0:            # Linear Regression
            return sk_LinearRegression(n_jobs=-1)

        if self._lasso_ratio == 0:      # Ridge Regression
            return SklearnRidge(alpha=self._alpha)

        if self._lasso_ratio == 1:      # Lasso Regression
            return SklearnLasso(alpha=self._alpha)

        return SklearnElasticNet(       # Elastic Net Regression
            alpha=self._alpha,
            l1_ratio=self._lasso_ratio,
        )

    def _get_models_for_all_choices(self) -> list[ElasticNetRegressor]:
        alpha_choices = self._alpha if isinstance(self._alpha, Choice) else [self._alpha]
        lasso_choices = self._lasso_ratio if isinstance(self._lasso_ratio, Choice) else [self._lasso_ratio]

        models = []
        for a in alpha_choices:
            for lasso in lasso_choices:
                models.append(ElasticNetRegressor(alpha=a, lasso_ratio=lasso))
        return models

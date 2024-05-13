from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from warnings import warn

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class ElasticNetRegressor(Regressor):
    """Elastic net regression.

    Parameters
    ----------
    alpha:
        Controls the regularization of the model. The higher the value, the more regularized it becomes.
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

    def __init__(self, *, alpha: float = 1.0, lasso_ratio: float = 0.5) -> None:
        super().__init__()

        # Validation
        _check_bounds("alpha", alpha, lower_bound=_ClosedBound(0))
        if alpha == 0:
            warn(
                (
                    "Setting alpha to zero makes this model equivalent to LinearRegression. You should use "
                    "LinearRegression instead for better numerical stability."
                ),
                UserWarning,
                stacklevel=2,
            )

        _check_bounds("lasso_ratio", lasso_ratio, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(1))
        if lasso_ratio == 0:
            warnings.warn(
                (
                    "ElasticNetRegression with lasso_ratio = 0 is essentially RidgeRegression."
                    " Use RidgeRegression instead for better numerical stability."
                ),
                stacklevel=2,
            )
        elif lasso_ratio == 1:
            warnings.warn(
                (
                    "ElasticNetRegression with lasso_ratio = 0 is essentially LassoRegression."
                    " Use LassoRegression instead for better numerical stability."
                ),
                stacklevel=2,
            )

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
    def alpha(self) -> float:
        """The regularization of the model."""
        return self._alpha

    @property
    def lasso_ratio(self) -> float:
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

        return SklearnElasticNet(
            alpha=self._alpha,
            l1_ratio=self._lasso_ratio,
        )

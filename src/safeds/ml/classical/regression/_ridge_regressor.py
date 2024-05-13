from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class RidgeRegressor(Regressor):
    """
    Ridge regression.

    Parameters
    ----------
    alpha:
        Controls the regularization of the model. The higher the value, the more regularized it becomes.

    Raises
    ------
    OutOfBoundsError
        If `alpha` is negative.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, *, alpha: float = 1.0) -> None:
        super().__init__()

        # Validation
        _check_bounds("alpha", alpha, lower_bound=_ClosedBound(0))
        if alpha == 0.0:
            warnings.warn(
                (
                    "Setting alpha to zero makes this model equivalent to LinearRegression. You should use "
                    "LinearRegression instead for better numerical stability."
                ),
                UserWarning,
                stacklevel=2,
            )

        # Hyperparameters
        self._alpha = alpha

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._alpha,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """
        Get the regularization of the model.

        Returns
        -------
        result:
            The regularization of the model.
        """
        return self._alpha

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> RidgeRegressor:
        return RidgeRegressor(alpha=self._alpha)

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.linear_model import Ridge as SklearnRidge

        return SklearnRidge(alpha=self._alpha)

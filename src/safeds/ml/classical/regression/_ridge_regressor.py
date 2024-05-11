from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


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
        if alpha < 0:
            raise OutOfBoundsError(alpha, name="alpha", lower_bound=ClosedBound(0))
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

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        pass

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> RidgeRegressor:
        return RidgeRegressor(alpha=self._alpha)

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.linear_model import Ridge as SklearnRidge

        return SklearnRidge(alpha=self._alpha)

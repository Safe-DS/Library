from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class LinearRegressor(Regressor):
    """Linear regression."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> LinearRegressor:
        return LinearRegressor()

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.linear_model import LinearRegression as sk_LinearRegression

        return sk_LinearRegression(n_jobs=-1)

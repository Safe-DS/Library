from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._bases import _GradientBoostingBase

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class GradientBoostingRegressor(Regressor, _GradientBoostingBase):
    """
    Gradient boosting regression.

    Parameters
    ----------
    tree_count:
        The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large
        number usually results in better performance.
    learning_rate:
        The larger the value, the more the model is influenced by each additional tree. If the learning rate is too
        low, the model might underfit. If the learning rate is too high, the model might overfit.

    Raises
    ------
    OutOfBoundsError
        If `tree_count` or `learning_rate` are less than or equal to 0.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        tree_count: int = 100,
        learning_rate: float = 0.1,
    ) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _GradientBoostingBase.__init__(
            self,
            tree_count=tree_count,
            learning_rate=learning_rate,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            _GradientBoostingBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            tree_count=self._tree_count,
            learning_rate=self._learning_rate,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoostingRegressor

        return SklearnGradientBoostingRegressor(
            n_estimators=self._tree_count,
            learning_rate=self._learning_rate,
        )

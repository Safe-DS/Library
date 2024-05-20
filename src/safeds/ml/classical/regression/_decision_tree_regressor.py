from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._bases import _DecisionTreeBase

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class DecisionTreeRegressor(Regressor, _DecisionTreeBase):
    """
    Decision tree regression.

    Parameters
    ----------
    max_depth:
        The maximum depth of each tree. If None, the depth is not limited. Has to be greater than 0.
    min_sample_count_in_leaves:
        The minimum number of samples that must remain in the leaves of each tree. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `max_depth` is less than 1.
    OutOfBoundsError
        If `min_sample_count_in_leaves` is less than 1.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_sample_count_in_leaves: int = 5,
    ) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _DecisionTreeBase.__init__(
            self,
            max_depth=max_depth,
            min_sample_count_in_leaves=min_sample_count_in_leaves,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            _DecisionTreeBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            max_depth=self._max_depth,
            min_sample_count_in_leaves=self._min_sample_count_in_leaves,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

        return SklearnDecisionTreeRegressor(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_sample_count_in_leaves,
        )

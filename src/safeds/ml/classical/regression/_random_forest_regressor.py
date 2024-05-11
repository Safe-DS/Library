from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _get_random_seed, _structural_hash
from safeds.ml.classical._bases import _RandomForestBase

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class RandomForestRegressor(Regressor, _RandomForestBase):
    """
    Random forest regression.

    Parameters
    ----------
    number_of_trees:
        The number of trees to be used in the random forest. Has to be greater than 0.
    maximum_depth:
        The maximum depth of each tree. If None, the depth is not limited. Has to be greater than 0.
    minimum_number_of_samples_in_leaves:
        The minimum number of samples that must remain in the leaves of each tree. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `number_of_trees` is less than 1.
    OutOfBoundsError
        If `maximum_depth` is less than 1.
    OutOfBoundsError
        If `minimum_number_of_samples_in_leaves` is less than 1.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        number_of_trees: int = 100,
        maximum_depth: int | None = None,
        minimum_number_of_samples_in_leaves: int = 5,
    ) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _RandomForestBase.__init__(
            self,
            number_of_trees=number_of_trees,
            maximum_depth=maximum_depth,
            minimum_number_of_samples_in_leaves=minimum_number_of_samples_in_leaves,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            _RandomForestBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            number_of_trees=self._number_of_trees,
            maximum_depth=self._maximum_depth,
            minimum_number_of_samples_in_leaves=self._minimum_number_of_samples_in_leaves,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

        return SklearnRandomForestRegressor(
            n_estimators=self._number_of_trees,
            max_depth=self._maximum_depth,
            min_samples_leaf=self._minimum_number_of_samples_in_leaves,
            random_state=_get_random_seed(),
            n_jobs=-1,
        )

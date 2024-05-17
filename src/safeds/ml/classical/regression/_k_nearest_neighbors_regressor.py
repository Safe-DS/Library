from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._bases import _KNearestNeighborsBase

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.labeled.containers import TabularDataset


class KNearestNeighborsRegressor(Regressor, _KNearestNeighborsBase):
    """
    K-nearest-neighbors regression.

    Parameters
    ----------
    neighbor_count:
        The number of neighbors to use for interpolation. Has to be greater than 0 (validated in the constructor) and
        less than or equal to the sample size (validated when calling `fit`).

    Raises
    ------
    OutOfBoundsError
        If `neighbor_count` is less than 1.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, neighbor_count: int) -> None:
        # Initialize superclasses
        Regressor.__init__(self)
        _KNearestNeighborsBase.__init__(
            self,
            neighbor_count=neighbor_count,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Regressor.__hash__(self),
            _KNearestNeighborsBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset) -> None:
        if self._neighbor_count > training_set.to_table().row_count:
            raise ValueError(
                (
                    f"The parameter 'neighbor_count' ({self._neighbor_count}) has to be less than or equal to"
                    f" the sample size ({training_set.to_table().row_count})."
                ),
            )

    def _clone(self) -> KNearestNeighborsRegressor:
        return KNearestNeighborsRegressor(
            neighbor_count=self._neighbor_count,
        )

    def _get_sklearn_model(self) -> RegressorMixin:
        from sklearn.neighbors import KNeighborsRegressor as SklearnKNeighborsRegressor

        return SklearnKNeighborsRegressor(
            n_neighbors=self._neighbor_count,
            n_jobs=-1,
        )

from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._bases import _KNearestNeighborsBase

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.labeled.containers import TabularDataset


class KNearestNeighborsClassifier(Classifier, _KNearestNeighborsBase):
    """
    K-nearest-neighbors classification.

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

    def __init__(
        self,
        neighbor_count: int,
    ) -> None:
        # Initialize superclasses
        Classifier.__init__(self)
        _KNearestNeighborsBase.__init__(
            self,
            neighbor_count=neighbor_count,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            _KNearestNeighborsBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset) -> None:
        if self._neighbor_count > training_set._table.row_count:
            raise ValueError(
                (
                    f"The parameter 'neighbor_count' ({self._neighbor_count}) has to be less than or equal to"
                    f" the sample size ({training_set._table.row_count})."
                ),
            )

    def _clone(self) -> KNearestNeighborsClassifier:
        return KNearestNeighborsClassifier(
            neighbor_count=self._neighbor_count,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier

        return SklearnKNeighborsClassifier(
            n_neighbors=self._neighbor_count,
            n_jobs=-1,
        )

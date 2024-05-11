from __future__ import annotations

from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class KNearestNeighborsClassifier(Classifier):
    """
    K-nearest-neighbors classification.

    Parameters
    ----------
    number_of_neighbors:
        The number of neighbors to use for interpolation. Has to be greater than 0 (validated in the constructor) and
        less than or equal to the sample size (validated when calling `fit`).

    Raises
    ------
    OutOfBoundsError
        If `number_of_neighbors` is less than 1.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, number_of_neighbors: int) -> None:
        super().__init__()

        # Validation
        if number_of_neighbors < 1:
            raise OutOfBoundsError(number_of_neighbors, name="number_of_neighbors", lower_bound=ClosedBound(1))

        # Hyperparameters
        self._number_of_neighbors = number_of_neighbors

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._number_of_neighbors,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def number_of_neighbors(self) -> int:
        """
        Get the number of neighbors used for interpolation.

        Returns
        -------
        result:
            The number of neighbors.
        """
        return self._number_of_neighbors

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        if self._number_of_neighbors > training_set._table.number_of_rows:
            raise ValueError(
                (
                    f"The parameter 'number_of_neighbors' ({self._number_of_neighbors}) has to be less than or equal to"
                    f" the sample size ({training_set._table.number_of_rows})."
                ),
            )

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> Self:
        return KNearestNeighborsClassifier(
            number_of_neighbors=self._number_of_neighbors,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier

        return SklearnKNeighborsClassifier(
            n_neighbors=self._number_of_neighbors,
            n_jobs=-1,
        )

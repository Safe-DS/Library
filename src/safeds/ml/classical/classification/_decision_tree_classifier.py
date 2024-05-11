from __future__ import annotations

from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class DecisionTreeClassifier(Classifier):
    """
    Decision tree classification.

    Parameters
    ----------
    maximum_depth:
        The maximum depth of each tree. If None, the depth is not limited. Has to be greater than 0.
    minimum_number_of_samples_in_leaves:
        The minimum number of samples that must remain in the leaves of each tree. Has to be greater than 0.

    Raises
    ------
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
        maximum_depth: int | None = None,
        minimum_number_of_samples_in_leaves: int = 1,
    ) -> None:
        super().__init__()

        # Validation
        if maximum_depth is not None and maximum_depth < 1:
            raise OutOfBoundsError(maximum_depth, name="maximum_depth", lower_bound=ClosedBound(1))
        if minimum_number_of_samples_in_leaves < 1:
            raise OutOfBoundsError(
                minimum_number_of_samples_in_leaves,
                name="minimum_number_of_samples_in_leaves",
                lower_bound=ClosedBound(1),
            )

        # Hyperparameters
        self._maximum_depth: int | None = maximum_depth
        self._minimum_number_of_samples_in_leaves: int = minimum_number_of_samples_in_leaves

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._maximum_depth,
            self._minimum_number_of_samples_in_leaves,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def maximum_depth(self) -> int | None:
        """The maximum depth of the tree."""
        return self._maximum_depth

    @property
    def minimum_number_of_samples_in_leaves(self) -> int:
        """The minimum number of samples that must remain in the leaves of the tree."""
        return self._minimum_number_of_samples_in_leaves

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        pass

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> Self:
        return DecisionTreeClassifier(
            maximum_depth=self._maximum_depth,
            minimum_number_of_samples_in_leaves=self._minimum_number_of_samples_in_leaves,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

        return SklearnDecisionTreeClassifier(
            max_depth=self._maximum_depth,
            min_samples_leaf=self._minimum_number_of_samples_in_leaves,
        )

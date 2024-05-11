from __future__ import annotations

from typing import TYPE_CHECKING, Self

from safeds._utils import _get_random_seed, _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class RandomForestClassifier(Classifier):
    """
    Random forest classification.

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
        minimum_number_of_samples_in_leaves: int = 1,
    ) -> None:
        super().__init__()

        # Validation
        if number_of_trees < 1:
            raise OutOfBoundsError(number_of_trees, name="number_of_trees", lower_bound=ClosedBound(1))
        if maximum_depth is not None and maximum_depth < 1:
            raise OutOfBoundsError(maximum_depth, name="maximum_depth", lower_bound=ClosedBound(1))
        if minimum_number_of_samples_in_leaves < 1:
            raise OutOfBoundsError(
                minimum_number_of_samples_in_leaves,
                name="minimum_number_of_samples_in_leaves",
                lower_bound=ClosedBound(1),
            )

        # Hyperparameters
        self._number_of_trees: int = number_of_trees
        self._maximum_depth: int | None = maximum_depth
        self._minimum_number_of_samples_in_leaves: int = minimum_number_of_samples_in_leaves

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            self._number_of_trees,
            self._maximum_depth,
            self._minimum_number_of_samples_in_leaves,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def number_of_trees(self) -> int:
        """The number of trees used in the random forest."""
        return self._number_of_trees

    @property
    def maximum_depth(self) -> int | None:
        """The maximum depth of each tree."""
        return self._maximum_depth

    @property
    def minimum_number_of_samples_in_leaves(self) -> int:
        """The minimum number of samples that must remain in the leaves of each tree."""
        return self._minimum_number_of_samples_in_leaves

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):
        pass

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):
        pass

    def _clone(self) -> Self:
        return RandomForestClassifier(
            number_of_trees=self._number_of_trees,
            maximum_depth=self._maximum_depth,
            minimum_number_of_samples_in_leaves=self._minimum_number_of_samples_in_leaves,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

        return SklearnRandomForestClassifier(
            n_estimators=self._number_of_trees,
            max_depth=self._maximum_depth,
            min_samples_leaf=self._minimum_number_of_samples_in_leaves,
            random_state=_get_random_seed(),
            n_jobs=-1,
        )

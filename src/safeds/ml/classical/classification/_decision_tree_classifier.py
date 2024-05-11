from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.ml.classical._bases import _DecisionTreeBase

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


class DecisionTreeClassifier(Classifier, _DecisionTreeBase):
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
        # Initialize superclasses
        Classifier.__init__(self)
        _DecisionTreeBase.__init__(
            self,
            maximum_depth=maximum_depth,
            minimum_number_of_samples_in_leaves=minimum_number_of_samples_in_leaves,
        )

    def __hash__(self) -> int:
        return _structural_hash(
            Classifier.__hash__(self),
            _DecisionTreeBase.__hash__(self),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _clone(self) -> DecisionTreeClassifier:
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

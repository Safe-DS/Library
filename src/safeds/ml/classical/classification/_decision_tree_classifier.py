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
        min_sample_count_in_leaves: int = 1,
    ) -> None:
        # Initialize superclasses
        Classifier.__init__(self)
        _DecisionTreeBase.__init__(
            self,
            max_depth=max_depth,
            min_sample_count_in_leaves=min_sample_count_in_leaves,
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
            max_depth=self._max_depth,
            min_sample_count_in_leaves=self._min_sample_count_in_leaves,
        )

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

        return SklearnDecisionTreeClassifier(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_sample_count_in_leaves,
        )

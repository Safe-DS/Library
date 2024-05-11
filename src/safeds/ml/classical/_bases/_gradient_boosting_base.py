from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError


class _GradientBoostingBase(ABC):

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        number_of_trees: int,
        learning_rate: float,
    ) -> None:
        # Validation
        if number_of_trees < 1:
            raise OutOfBoundsError(number_of_trees, name="number_of_trees", lower_bound=ClosedBound(1))
        if learning_rate <= 0:
            raise OutOfBoundsError(learning_rate, name="learning_rate", lower_bound=OpenBound(0))

        # Hyperparameters
        self._number_of_trees = number_of_trees
        self._learning_rate = learning_rate

    def __hash__(self) -> int:
        return _structural_hash(
            self._number_of_trees,
            self._learning_rate,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def number_of_trees(self) -> int:
        """The number of trees (estimators) in the ensemble."""
        return self._number_of_trees

    @property
    def learning_rate(self) -> float:
        """The learning rate."""
        return self._learning_rate

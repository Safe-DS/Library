from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound, _OpenBound


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
        _check_bounds("number_of_trees", number_of_trees, lower_bound=_ClosedBound(1))
        _check_bounds("learning_rate", learning_rate, lower_bound=_OpenBound(0))

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

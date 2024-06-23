from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound, _OpenBound
from safeds.ml.hyperparameters import Choice


class _GradientBoostingBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        tree_count: int | Choice[int],
        learning_rate: float | Choice[float],
    ) -> None:
        # Validation
        if isinstance(tree_count, Choice):
            for tc in tree_count:
                _check_bounds("tree_count", tc, lower_bound=_ClosedBound(1))
        else:
            _check_bounds("tree_count", tree_count, lower_bound=_ClosedBound(1))

        if isinstance(learning_rate, Choice):
            for lr in learning_rate:
                _check_bounds("learning_rate", lr, lower_bound=_OpenBound(0))
        else:
            _check_bounds("learning_rate", learning_rate, lower_bound=_OpenBound(0))

        # Hyperparameters
        self._tree_count = tree_count
        self._learning_rate = learning_rate

    def __hash__(self) -> int:
        return _structural_hash(
            self._tree_count,
            self._learning_rate,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def tree_count(self) -> int | Choice[int]:
        """The number of trees (estimators) in the ensemble."""
        return self._tree_count

    @property
    def learning_rate(self) -> float | Choice[float]:
        """The learning rate."""
        return self._learning_rate

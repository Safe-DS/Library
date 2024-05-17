from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound


class _DecisionTreeBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        max_depth: int | None,
        min_sample_count_in_leaves: int,
    ) -> None:
        # Validation
        _check_bounds("max_depth", max_depth, lower_bound=_ClosedBound(1))
        _check_bounds(
            "min_sample_count_in_leaves",
            min_sample_count_in_leaves,
            lower_bound=_ClosedBound(1),
        )

        # Hyperparameters
        self._max_depth: int | None = max_depth
        self._min_sample_count_in_leaves: int = min_sample_count_in_leaves

    def __hash__(self) -> int:
        return _structural_hash(
            self._max_depth,
            self._min_sample_count_in_leaves,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def max_depth(self) -> int | None:
        """The maximum depth of the tree."""
        return self._max_depth

    @property
    def min_sample_count_in_leaves(self) -> int:
        """The minimum number of samples that must remain in the leaves of the tree."""
        return self._min_sample_count_in_leaves

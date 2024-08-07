from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.ml.hyperparameters import Choice


class _RandomForestBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        tree_count: int | Choice[int],
        max_depth: int | None | Choice[int | None],
        min_sample_count_in_leaves: int | Choice[int],
    ) -> None:
        # Validation
        if isinstance(tree_count, Choice):
            for tc in tree_count:
                _check_bounds("tree_count", tc, lower_bound=_ClosedBound(1))
        else:
            _check_bounds("tree_count", tree_count, lower_bound=_ClosedBound(1))

        if isinstance(max_depth, Choice):
            for md in max_depth:
                _check_bounds("max_depth", md, lower_bound=_ClosedBound(1))
        else:
            _check_bounds("max_depth", max_depth, lower_bound=_ClosedBound(1))

        if isinstance(min_sample_count_in_leaves, Choice):
            for msc in min_sample_count_in_leaves:
                _check_bounds("min_sample_count_in_leaves", msc, lower_bound=_ClosedBound(1))
        else:
            _check_bounds("min_sample_count_in_leaves", min_sample_count_in_leaves, lower_bound=_ClosedBound(1))

        # Hyperparameters
        self._tree_count: int | Choice[int] = tree_count
        self._max_depth: int | None | Choice[int | None] = max_depth
        self._min_sample_count_in_leaves: int | Choice[int] = min_sample_count_in_leaves

    def __hash__(self) -> int:
        return _structural_hash(
            self._tree_count,
            self._max_depth,
            self._min_sample_count_in_leaves,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def tree_count(self) -> int | Choice[int]:
        """The number of trees used in the random forest."""
        return self._tree_count

    @property
    def max_depth(self) -> int | None | Choice[int | None]:
        """The maximum depth of each tree."""
        return self._max_depth

    @property
    def min_sample_count_in_leaves(self) -> int | Choice[int]:
        """The minimum number of samples that must remain in the leaves of each tree."""
        return self._min_sample_count_in_leaves

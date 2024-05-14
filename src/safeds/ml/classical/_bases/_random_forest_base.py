from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound


class _RandomForestBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        number_of_trees: int,
        maximum_depth: int | None,
        minimum_number_of_samples_in_leaves: int,
    ) -> None:
        # Validation
        _check_bounds("number_of_trees", number_of_trees, lower_bound=_ClosedBound(1))
        _check_bounds("maximum_depth", maximum_depth, lower_bound=_ClosedBound(1))
        _check_bounds(
            "minimum_number_of_samples_in_leaves",
            minimum_number_of_samples_in_leaves,
            lower_bound=_ClosedBound(1),
        )

        # Hyperparameters
        self._number_of_trees: int = number_of_trees
        self._maximum_depth: int | None = maximum_depth
        self._minimum_number_of_samples_in_leaves: int = minimum_number_of_samples_in_leaves

    def __hash__(self) -> int:
        return _structural_hash(
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

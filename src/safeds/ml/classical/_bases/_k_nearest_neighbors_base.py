from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound


class _KNearestNeighborsBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        number_of_neighbors: int,
    ) -> None:
        # Validation
        _check_bounds("number_of_neighbors", number_of_neighbors, lower_bound=_ClosedBound(1))

        # Hyperparameters
        self._number_of_neighbors = number_of_neighbors

    def __hash__(self) -> int:
        return _structural_hash(
            self._number_of_neighbors,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def number_of_neighbors(self) -> int:
        """The number of neighbors used for interpolation."""
        return self._number_of_neighbors

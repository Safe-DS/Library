from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError


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
        if number_of_neighbors < 1:
            raise OutOfBoundsError(number_of_neighbors, name="number_of_neighbors", lower_bound=ClosedBound(1))

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

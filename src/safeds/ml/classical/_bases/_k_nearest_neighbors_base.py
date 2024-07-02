from __future__ import annotations

from abc import ABC, abstractmethod

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.ml.hyperparameters import Choice


class _KNearestNeighborsBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        neighbor_count: int | Choice[int],
    ) -> None:
        # Validation
        if isinstance(neighbor_count, Choice):
            for nc in neighbor_count:
                _check_bounds("neighbor_count", nc, lower_bound=_ClosedBound(1))
        else:
            _check_bounds("neighbor_count", neighbor_count, lower_bound=_ClosedBound(1))

        # Hyperparameters
        self._neighbor_count = neighbor_count

    def __hash__(self) -> int:
        return _structural_hash(
            self._neighbor_count,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def neighbor_count(self) -> int | Choice[int]:
        """The number of neighbors used for interpolation."""
        return self._neighbor_count

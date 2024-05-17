from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound, _OpenBound

if TYPE_CHECKING:
    from safeds.ml.classical import SupervisedModel


class _AdaBoostBase(ABC):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        max_learner_count: int,
        learning_rate: float,
    ) -> None:
        # Validation
        _check_bounds("max_learner_count", max_learner_count, lower_bound=_ClosedBound(1))
        _check_bounds("learning_rate", learning_rate, lower_bound=_OpenBound(0))

        # Hyperparameters
        self._max_learner_count: int = max_learner_count
        self._learning_rate: float = learning_rate

    def __hash__(self) -> int:
        return _structural_hash(
            self._max_learner_count,
            self._learning_rate,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def max_learner_count(self) -> int:
        """The maximum number of learners in the ensemble."""
        return self._max_learner_count

    @property
    def learning_rate(self) -> float:
        """The learning rate."""
        return self._learning_rate

    @property
    @abstractmethod
    def learner(self) -> SupervisedModel | None:
        """The base learner used for training the ensemble."""

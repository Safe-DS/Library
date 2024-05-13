from __future__ import annotations

from abc import ABC, abstractmethod


class Dataset(ABC):
    """A dataset is used as input to machine learning models."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

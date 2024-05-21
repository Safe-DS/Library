from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

In_co = TypeVar("In_co", covariant=True)
Out_co = TypeVar("Out_co", covariant=True)


class Dataset(Generic[In_co, Out_co], ABC):
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

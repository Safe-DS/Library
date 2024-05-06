from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

T = TypeVar("T")


class ExperimentalPolarsCell(ABC, Generic[T]):
    pass

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


# TODO: should not be abstract
class ExperimentalPolarsColumn(ABC, Sequence[T]):
    pass

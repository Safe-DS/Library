from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TypeVar

from safeds.data.tabular.containers import ExperimentalPolarsCell

T = TypeVar("T")


# TODO: should not be abstract
class ExperimentalPolarsColumn(ExperimentalPolarsCell[T], Sequence[T], ABC):
    pass

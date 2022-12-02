from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ColumnType(ABC):
    """
    Base Type for Columns, stored in TableSchema
    """

    @abstractmethod
    def is_numeric(self) -> bool:
        """
        tells if the given column type is numeric
        Returns
        -------
        bool
        """
        return False

    @staticmethod
    def from_numpy_dtype(_type: np.dtype) -> ColumnType:
        """
        return the column type for a given numpy dtype
        Parameters
        ----------
        _type : numpy.dtype

        Returns
        -------
        ColumnType

        """
        if _type.kind in ("u", "i"):
            return IntColumnType()
        if _type.kind == "b":
            return BooleanColumnType()
        if _type.kind == "f":
            return FloatColumnType()
        if _type.kind in ("S", "U", "O"):
            return StringColumnType()
        raise TypeError("Unexpected column type")


@dataclass
class IntColumnType(ColumnType):
    def is_numeric(self) -> bool:
        return True

    def __repr__(self):
        return "int"


@dataclass
class BooleanColumnType(ColumnType):
    def is_numeric(self) -> bool:
        return False

    def __repr__(self):
        return "bool"


@dataclass
class FloatColumnType(ColumnType):
    def is_numeric(self) -> bool:
        return True

    def __repr__(self):
        return "float"


@dataclass
class StringColumnType(ColumnType):
    def is_numeric(self) -> bool:
        return False

    def __repr__(self):
        return "string"


@dataclass
class OptionalColumnType(ColumnType):
    _type: ColumnType

    def is_numeric(self) -> bool:
        return self._type.is_numeric()

    def __repr__(self):
        return f"optional({self._type.__repr__()})"

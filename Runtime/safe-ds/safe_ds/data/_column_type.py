from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ColumnType(ABC):
    """
    Base type for Column, stored in TableSchema.
    """

    @abstractmethod
    def is_numeric(self) -> bool:
        """
        Return whether the given column type is numeric.
        Returns
        -------
        is_numeric : bool
            True if the column is numeric.
        """
        return False

    @staticmethod
    def from_numpy_dtype(_type: np.dtype) -> ColumnType:
        """
        Return the column type for a given numpy dtype.
        Parameters
        ----------
        _type : numpy.dtype

        Returns
        -------
        column_type : ColumnType
            The ColumnType.

        Raises
        -------
        TypeError
            If an unexpected column type is parsed.

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

    def __repr__(self) -> str:
        return "int"


@dataclass
class BooleanColumnType(ColumnType):
    def is_numeric(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "bool"


@dataclass
class FloatColumnType(ColumnType):
    def is_numeric(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "float"


@dataclass
class StringColumnType(ColumnType):
    def is_numeric(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "string"


@dataclass
class OptionalColumnType(ColumnType):
    _type: ColumnType

    def is_numeric(self) -> bool:
        return self._type.is_numeric()

    def __repr__(self) -> str:
        return f"optional({self._type.__repr__()})"

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ColumnType(ABC):
    """
    Abstract base class for column types.
    """

    @abstractmethod
    def is_nullable(self) -> bool:
        """
        Return whether the given column type is nullable.

        Returns
        -------
        is_nullable : bool
            True if the column is nullable.
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

    @staticmethod
    def _from_numpy_dtype(dtype: np.dtype) -> ColumnType:
        """
        Return the column type for a given numpy dtype.

        Parameters
        ----------
        dtype : numpy.dtype
            The numpy dtype.

        Returns
        -------
        column_type : ColumnType
            The ColumnType.

        Raises
        -------
        TypeError
            If the given dtype is not supported.
        """
        if dtype.kind in ("u", "i"):
            return Int()
        if dtype.kind == "b":
            return Boolean()
        if dtype.kind == "f":
            return Float()
        if dtype.kind in ("S", "U", "O"):
            return String()
        raise TypeError("Unexpected column type")


@dataclass
class Mixed(ColumnType):
    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "Mixed"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        return self._is_nullable

    def is_numeric(self) -> bool:
        return False


@dataclass
class Boolean(ColumnType):
    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "Boolean"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        return self._is_nullable

    def is_numeric(self) -> bool:
        return False


@dataclass
class Float(ColumnType):
    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "Float"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        return self._is_nullable

    def is_numeric(self) -> bool:
        return True


@dataclass
class Int(ColumnType):
    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "Int"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        return self._is_nullable

    def is_numeric(self) -> bool:
        return True


@dataclass
class String(ColumnType):
    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "String"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        return self._is_nullable

    def is_numeric(self) -> bool:
        return False

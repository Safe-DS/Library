from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ColumnType(ABC):
    """Abstract base class for column types."""

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
        ------
        TypeError
            If the given dtype is not supported.
        """
        if dtype.kind in ("u", "i"):
            return Integer()
        if dtype.kind == "b":
            return Boolean()
        if dtype.kind == "f":
            return RealNumber()
        if dtype.kind in ("S", "U", "O", "M", "m"):
            return String()
        raise TypeError("Unexpected column type")


@dataclass
class Anything(ColumnType):
    """
    Type for a column that contains anything.

    Parameters
    ----------
    is_nullable : bool
        Whether the type also allows null values.
    """

    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "Anything"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        """
        Return whether the given column type is nullable.

        Returns
        -------
        is_nullable : bool
            True if the column is nullable.
        """
        return self._is_nullable

    def is_numeric(self) -> bool:
        """
        Return whether the given column type is numeric.

        Returns
        -------
        is_numeric : bool
            True if the column is numeric.
        """
        return False


@dataclass
class Boolean(ColumnType):
    """
    Type for a column that only contains booleans.

    Parameters
    ----------
    is_nullable : bool
        Whether the type also allows null values.
    """

    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "Boolean"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        """
        Return whether the given column type is nullable.

        Returns
        -------
        is_nullable : bool
            True if the column is nullable.
        """
        return self._is_nullable

    def is_numeric(self) -> bool:
        """
        Return whether the given column type is numeric.

        Returns
        -------
        is_numeric : bool
            True if the column is numeric.
        """
        return False


@dataclass
class RealNumber(ColumnType):
    """
    Type for a column that only contains real numbers.

    Parameters
    ----------
    is_nullable : bool
        Whether the type also allows null values.
    """

    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "RealNumber"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        """
        Return whether the given column type is nullable.

        Returns
        -------
        is_nullable : bool
            True if the column is nullable.
        """
        return self._is_nullable

    def is_numeric(self) -> bool:
        """
        Return whether the given column type is numeric.

        Returns
        -------
        is_numeric : bool
            True if the column is numeric.
        """
        return True


@dataclass
class Integer(ColumnType):
    """
    Type for a column that only contains integers.

    Parameters
    ----------
    is_nullable : bool
        Whether the type also allows null values.
    """

    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "Integer"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        """
        Return whether the given column type is nullable.

        Returns
        -------
        is_nullable : bool
            True if the column is nullable.
        """
        return self._is_nullable

    def is_numeric(self) -> bool:
        """
        Return whether the given column type is numeric.

        Returns
        -------
        is_numeric : bool
            True if the column is numeric.
        """
        return True


@dataclass
class String(ColumnType):
    """
    Type for a column that only contains strings.

    Parameters
    ----------
    is_nullable : bool
        Whether the type also allows null values.
    """

    _is_nullable: bool

    def __init__(self, is_nullable: bool = False):
        self._is_nullable = is_nullable

    def __repr__(self) -> str:
        result = "String"
        if self._is_nullable:
            result += "?"
        return result

    def is_nullable(self) -> bool:
        """
        Return whether the given column type is nullable.

        Returns
        -------
        is_nullable : bool
            True if the column is nullable.
        """
        return self._is_nullable

    def is_numeric(self) -> bool:
        """
        Return whether the given column type is numeric.

        Returns
        -------
        is_numeric : bool
            True if the column is numeric.
        """
        return False

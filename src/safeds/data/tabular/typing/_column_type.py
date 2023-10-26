from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import NoneType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class ColumnType(ABC):
    """Abstract base class for column types."""

    _is_nullable: bool  # This line is just here so the linter doesn't throw an error

    @abstractmethod
    def __init__(self, is_nullable: bool = False) -> None:
        """
        Abstract initializer for ColumnType.

        Parameters
        ----------
        is_nullable
            Whether the columntype is nullable.
        """

    @staticmethod
    def _data_type(data: pd.Series) -> ColumnType:
        """
        Return the column type for a given `Series` from `pandas`.

        Parameters
        ----------
        data : pd.Series
            The data to be checked.

        Returns
        -------
        column_type : ColumnType
            The ColumnType.

        Raises
        ------
        NotImplementedError
            If the given data type is not supported.
        """

        def column_type_of_type(cell_type: Any) -> ColumnType:
            if cell_type == int or cell_type == np.int64 or cell_type == np.int32:
                return Integer(is_nullable)
            if cell_type == float or cell_type == np.float64 or cell_type == np.float32:
                return RealNumber(is_nullable)
            if cell_type == bool:
                return Boolean(is_nullable)
            if cell_type == str:
                return String(is_nullable)
            if cell_type is NoneType:
                return Nothing()
            else:
                message = f"Unsupported numpy data type '{cell_type}'."
                raise NotImplementedError(message)

        result: ColumnType = Nothing()
        is_nullable = False
        for cell in data:
            if result == Nothing():
                result = column_type_of_type(type(cell))
                if type(cell) is NoneType:
                    is_nullable = True
                    result._is_nullable = is_nullable
            if result != column_type_of_type(type(cell)):
                if type(cell) is NoneType:
                    is_nullable = True
                    result._is_nullable = is_nullable
                elif (isinstance(result, Integer) and isinstance(column_type_of_type(type(cell)), RealNumber)) or (
                    isinstance(result, RealNumber) and isinstance(column_type_of_type(type(cell)), Integer)
                ):
                    result = RealNumber(is_nullable)
                else:
                    result = Anything(is_nullable)
            if isinstance(cell, float) and np.isnan(cell):
                is_nullable = True
                result._is_nullable = is_nullable

        if isinstance(result, RealNumber) and all(
            data.apply(lambda c: bool(isinstance(c, float) and np.isnan(c) or c == float(int(c)))),
        ):
            result = Integer(is_nullable)

        return result

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

    def __init__(self, is_nullable: bool = False) -> None:
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

    def __init__(self, is_nullable: bool = False) -> None:
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

    def __init__(self, is_nullable: bool = False) -> None:
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

    def __init__(self, is_nullable: bool = False) -> None:
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

    def __init__(self, is_nullable: bool = False) -> None:
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


@dataclass
class Nothing(ColumnType):
    """Type for a column that contains None Values only."""

    _is_nullable: bool

    def __init__(self) -> None:
        self._is_nullable = True

    def __repr__(self) -> str:
        result = "Nothing"
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
        return True

    def is_numeric(self) -> bool:
        """
        Return whether the given column type is numeric.

        Returns
        -------
        is_numeric : bool
            True if the column is numeric.
        """
        return False

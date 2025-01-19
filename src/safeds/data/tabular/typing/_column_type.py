from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._validation import _check_time_zone

if TYPE_CHECKING:
    import polars as pl


class ColumnType(ABC):
    """
    The type of a column in a table.

    Use the static factory methods to create instances of this class.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------------------------------------------------------

    # Float --------------------------------------------------------------------

    @staticmethod
    def float32() -> ColumnType:
        """Create a `float32` column type (32-bit floating point number)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Float32())

    @staticmethod
    def float64() -> ColumnType:
        """Create a `float64` column type (64-bit floating point number)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Float64())

    # Signed int ---------------------------------------------------------------

    @staticmethod
    def int8() -> ColumnType:
        """Create an `int8` column type (8-bit signed integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Int8())

    @staticmethod
    def int16() -> ColumnType:
        """Create an `int16` column type (16-bit signed integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Int16())

    @staticmethod
    def int32() -> ColumnType:
        """Create an `int32` column type (32-bit signed integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Int32())

    @staticmethod
    def int64() -> ColumnType:
        """Create an `int64` column type (64-bit signed integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Int64())

    # Unsigned int -------------------------------------------------------------

    @staticmethod
    def uint8() -> ColumnType:
        """Create a `uint8` column type (8-bit unsigned integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.UInt8())

    @staticmethod
    def uint16() -> ColumnType:
        """Create a `uint16` column type (16-bit unsigned integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.UInt16())

    @staticmethod
    def uint32() -> ColumnType:
        """Create a `uint32` column type (32-bit unsigned integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.UInt32())

    @staticmethod
    def uint64() -> ColumnType:
        """Create a `uint64` column type (64-bit unsigned integer)."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.UInt64())

    # Temporal -----------------------------------------------------------------

    @staticmethod
    def date() -> ColumnType:
        """Create a `date` column type."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Date())

    @staticmethod
    def datetime(*, time_zone: str | None = None) -> ColumnType:
        """
        Create a `datetime` column type.

        Parameters
        ----------
        time_zone:
            The time zone. If None, values are assumed to be in local time. This is different from setting the time zone
            to `"UTC"`. Any TZ identifier defined in the
            [tz database](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) is valid.
        """
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        _check_time_zone(time_zone)

        return _PolarsColumnType(pl.Datetime(time_zone=time_zone))

    @staticmethod
    def duration() -> ColumnType:
        """Create a `duration` column type."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Duration())

    @staticmethod
    def time() -> ColumnType:
        """Create a `time` column type."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Time())

    # String -------------------------------------------------------------------

    @staticmethod
    def string() -> ColumnType:
        """Create a `string` column type."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.String())

    # Other --------------------------------------------------------------------

    @staticmethod
    def binary() -> ColumnType:
        """Create a `binary` column type."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Binary())

    @staticmethod
    def boolean() -> ColumnType:
        """Create a `boolean` column type."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Boolean())

    @staticmethod
    def null() -> ColumnType:
        """Create a `null` column type."""
        import polars as pl

        from ._polars_column_type import _PolarsColumnType  # circular import

        return _PolarsColumnType(pl.Null())

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def is_float(self) -> bool:
        """
        Whether the column type is a floating point type.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType
        >>> ColumnType.float32().is_float
        True

        >>> ColumnType.int8().is_float
        False
        """

    @property
    @abstractmethod
    def is_int(self) -> bool:
        """
        Whether the column type is an integer type (signed or unsigned).

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType
        >>> ColumnType.int8().is_int
        True

        >>> ColumnType.float32().is_int
        False
        """

    @property
    @abstractmethod
    def is_numeric(self) -> bool:
        """
        Whether the column type is a numeric type.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType
        >>> ColumnType.float32().is_numeric
        True

        >>> ColumnType.string().is_numeric
        False
        """

    @property
    @abstractmethod
    def is_signed_int(self) -> bool:
        """
        Whether the column type is a signed integer type.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType
        >>> ColumnType.int8().is_signed_int
        True

        >>> ColumnType.uint8().is_signed_int
        False
        """

    @property
    @abstractmethod
    def is_temporal(self) -> bool:
        """
        Whether the column type is a temporal type.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType
        >>> ColumnType.date().is_temporal
        True

        >>> ColumnType.string().is_temporal
        False
        """

    @property
    @abstractmethod
    def is_unsigned_int(self) -> bool:
        """
        Whether the column type is an unsigned integer type.

        Examples
        --------
        >>> from safeds.data.tabular.typing import ColumnType
        >>> ColumnType.uint8().is_unsigned_int
        True

        >>> ColumnType.int8().is_unsigned_int
        False
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def _polars_data_type(self) -> pl.DataType:
        """The Polars expression that corresponds to this cell."""

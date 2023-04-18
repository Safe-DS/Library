from polars import PolarsDataType, INTEGER_DTYPES as POLARS_INTEGER_DTYPES, FLOAT_DTYPES as POLARS_FLOAT_DTYPES
from polars.datatypes import Boolean as PolarsBoolean, Decimal as PolarsDecimal, Utf8 as PolarsUtf8

import numpy as np
import pytest
from safeds.data.tabular.typing import (
    Anything,
    Boolean,
    ColumnType,
    Integer,
    RealNumber,
    String,
)


class TestColumnType:
    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            (Anything(is_nullable=False), "Anything"),
            (Anything(is_nullable=True), "Anything?"),
            (Boolean(is_nullable=False), "Boolean"),
            (Boolean(is_nullable=True), "Boolean?"),
            (RealNumber(is_nullable=False), "RealNumber"),
            (RealNumber(is_nullable=True), "RealNumber?"),
            (Integer(is_nullable=False), "Integer"),
            (Integer(is_nullable=True), "Integer?"),
            (String(is_nullable=False), "String"),
            (String(is_nullable=True), "String?"),
        ],
        ids=repr,
    )
    def test_repr(self, column_type: ColumnType, expected: str) -> None:
        assert repr(column_type) == expected

    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            (Anything(is_nullable=False), False),
            (Anything(is_nullable=True), True),
            (Boolean(is_nullable=False), False),
            (Boolean(is_nullable=True), True),
            (RealNumber(is_nullable=False), False),
            (RealNumber(is_nullable=True), True),
            (Integer(is_nullable=False), False),
            (Integer(is_nullable=True), True),
            (String(is_nullable=False), False),
            (String(is_nullable=True), True),
        ],
        ids=repr,
    )
    def test_is_nullable(self, column_type: ColumnType, expected: bool) -> None:
        assert column_type.is_nullable() == expected

    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            (Anything(is_nullable=False), False),
            (Anything(is_nullable=True), False),
            (Boolean(is_nullable=False), False),
            (Boolean(is_nullable=True), False),
            (RealNumber(is_nullable=False), True),
            (RealNumber(is_nullable=True), True),
            (Integer(is_nullable=False), True),
            (Integer(is_nullable=True), True),
            (String(is_nullable=False), False),
            (String(is_nullable=True), False),
        ],
        ids=repr,
    )
    def test_is_numeric(self, column_type: ColumnType, expected: bool) -> None:
        assert column_type.is_numeric() == expected

    # Test cases taken from https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            # Boolean
            (np.dtype(np.bool_), Boolean()),
            # Number
            (np.dtype(np.half), RealNumber()),
            (np.dtype(np.single), RealNumber()),
            (np.dtype(np.float_), RealNumber()),
            (np.dtype(np.longfloat), RealNumber()),
            # Int
            (np.dtype(np.byte), Integer()),
            (np.dtype(np.short), Integer()),
            (np.dtype(np.intc), Integer()),
            (np.dtype(np.int_), Integer()),
            (np.dtype(np.longlong), Integer()),
            (np.dtype(np.ubyte), Integer()),
            (np.dtype(np.ushort), Integer()),
            (np.dtype(np.uintc), Integer()),
            (np.dtype(np.uint), Integer()),
            (np.dtype(np.ulonglong), Integer()),
            # String
            (np.dtype(np.str_), String()),
            (np.dtype(np.unicode_), String()),
            (np.dtype(np.object_), String()),
            (np.dtype(np.datetime64), String()),
            (np.dtype(np.timedelta64), String()),
        ],
        ids=repr,
    )
    def test_from_numpy_data_type(self, dtype: np.dtype, expected: ColumnType) -> None:
        assert ColumnType._from_numpy_data_type(dtype) == expected

    @pytest.mark.parametrize(
        ("data_type", "expected"),
        [
            # Boolean
            (PolarsBoolean, Boolean()),

            # Float
            *(
                (data_type, RealNumber())
                for data_type in POLARS_FLOAT_DTYPES
            ),
            (PolarsDecimal, RealNumber()),

            # Int
            *(
                (data_type, Integer())
                for data_type in POLARS_INTEGER_DTYPES
            ),

            # String
            (PolarsUtf8, String()),
        ],
        ids=repr,
    )
    def test_from_polars_data_type(self, data_type: PolarsDataType, expected: ColumnType) -> None:
        assert ColumnType._from_polars_data_type(data_type) == expected

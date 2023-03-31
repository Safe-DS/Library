import numpy as np
import pytest

from safeds.data.tabular.typing import ColumnType, Mixed, Boolean, String, Integer, Real


class TestColumnType:

    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            (Mixed(is_nullable=False), "Mixed"),
            (Mixed(is_nullable=True), "Mixed?"),
            (Boolean(is_nullable=False), "Boolean"),
            (Boolean(is_nullable=True), "Boolean?"),
            (Real(is_nullable=False), "Real"),
            (Real(is_nullable=True), "Real?"),
            (Integer(is_nullable=False), "Int"),
            (Integer(is_nullable=True), "Int?"),
            (String(is_nullable=False), "String"),
            (String(is_nullable=True), "String?"),
        ],
        ids=lambda x: repr(x)
    )
    def test_repr(self, column_type: ColumnType, expected: str):
        assert repr(column_type) == expected

    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            (Mixed(is_nullable=False), False),
            (Mixed(is_nullable=True), True),
            (Boolean(is_nullable=False), False),
            (Boolean(is_nullable=True), True),
            (Real(is_nullable=False), False),
            (Real(is_nullable=True), True),
            (Integer(is_nullable=False), False),
            (Integer(is_nullable=True), True),
            (String(is_nullable=False), False),
            (String(is_nullable=True), True),
        ],
        ids=lambda x: repr(x)
    )
    def test_is_nullable(self, column_type: ColumnType, expected: bool):
        assert column_type.is_nullable() == expected

    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            (Mixed(is_nullable=False), False),
            (Mixed(is_nullable=True), False),
            (Boolean(is_nullable=False), False),
            (Boolean(is_nullable=True), False),
            (Real(is_nullable=False), True),
            (Real(is_nullable=True), True),
            (Integer(is_nullable=False), True),
            (Integer(is_nullable=True), True),
            (String(is_nullable=False), False),
            (String(is_nullable=True), False),
        ],
        ids=lambda x: repr(x)
    )
    def test_is_numeric(self, column_type: ColumnType, expected: bool):
        assert column_type.is_numeric() == expected

    # Test cases taken from https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            # Boolean
            (np.dtype(np.bool_), Boolean()),

            # Number
            (np.dtype(np.half), Real()),
            (np.dtype(np.single), Real()),
            (np.dtype(np.float_), Real()),
            (np.dtype(np.longfloat), Real()),

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
        ids=lambda x: repr(x)
    )
    def test_from_numpy_dtype(self, dtype: np.dtype, expected: ColumnType):
        assert ColumnType._from_numpy_dtype(dtype) == expected

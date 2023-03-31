import numpy as np
import pytest

from safeds.data.tabular.typing import ColumnType, Mixed, Boolean, String, Int, Number


class TestColumnType:

    @pytest.mark.parametrize(
        ("column_type", "expected"),
        [
            (Mixed(is_nullable=False), "Mixed"),
            (Mixed(is_nullable=True), "Mixed?"),
            (Boolean(is_nullable=False), "Boolean"),
            (Boolean(is_nullable=True), "Boolean?"),
            (Number(is_nullable=False), "Number"),
            (Number(is_nullable=True), "Number?"),
            (Int(is_nullable=False), "Int"),
            (Int(is_nullable=True), "Int?"),
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
            (Number(is_nullable=False), False),
            (Number(is_nullable=True), True),
            (Int(is_nullable=False), False),
            (Int(is_nullable=True), True),
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
            (Number(is_nullable=False), True),
            (Number(is_nullable=True), True),
            (Int(is_nullable=False), True),
            (Int(is_nullable=True), True),
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
            (np.dtype(np.half), Number()),
            (np.dtype(np.single), Number()),
            (np.dtype(np.float_), Number()),
            (np.dtype(np.longfloat), Number()),

            # Int
            (np.dtype(np.byte), Int()),
            (np.dtype(np.short), Int()),
            (np.dtype(np.intc), Int()),
            (np.dtype(np.int_), Int()),
            (np.dtype(np.longlong), Int()),
            (np.dtype(np.ubyte), Int()),
            (np.dtype(np.ushort), Int()),
            (np.dtype(np.uintc), Int()),
            (np.dtype(np.uint), Int()),
            (np.dtype(np.ulonglong), Int()),

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

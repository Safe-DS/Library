import numpy as np
import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import (
    Anything,
    Boolean,
    ColumnType,
    Integer,
    RealNumber,
    String,
)


# class TestFromNumpyDataType:
#     # Test cases taken from https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
#     @pytest.mark.parametrize(
#         ("data_type", "expected"),
#         [
#             # Boolean
#             (np.dtype(np.bool_), Boolean()),
#             # Number
#             (np.dtype(np.half), RealNumber()),
#             (np.dtype(np.single), RealNumber()),
#             (np.dtype(np.float_), RealNumber()),
#             (np.dtype(np.longfloat), RealNumber()),
#             # Int
#             (np.dtype(np.byte), Integer()),
#             (np.dtype(np.short), Integer()),
#             (np.dtype(np.intc), Integer()),
#             (np.dtype(np.int_), Integer()),
#             (np.dtype(np.longlong), Integer()),
#             (np.dtype(np.ubyte), Integer()),
#             (np.dtype(np.ushort), Integer()),
#             (np.dtype(np.uintc), Integer()),
#             (np.dtype(np.uint), Integer()),
#             (np.dtype(np.ulonglong), Integer()),
#             # String
#             (np.dtype(np.str_), String()),
#             (np.dtype(np.unicode_), String()),
#             (np.dtype(np.object_), String()),
#             (np.dtype(np.datetime64), String()),
#             (np.dtype(np.timedelta64), String()),
#         ],
#         ids=repr,
#     )
#     def test_should_create_column_type_from_numpy_data_type(self, data_type: np.dtype, expected: ColumnType) -> None:
#         assert ColumnType._from_numpy_data_type(data_type) == expected
#
#     def test_should_raise_if_data_type_is_not_supported(self) -> None:
#         with pytest.raises(NotImplementedError):
#             ColumnType._from_numpy_data_type(np.dtype(np.void))
#

class TestDataType:
    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            (Column("a", [1, 2, 3]), Integer()),
            (Column("a", [1.0, 2.0, 3.0]), RealNumber()),
            (Column("a", [True, False, True]), Boolean()),
            (Column("a", ["a", "b", "c"]), String()),
            (Column("a", [None, None, None]), Anything(is_nullable=True)),
            (Column("a", [1, 2, None]), Anything(is_nullable=True)),
            (Column("a", [1.0, 2.0, None]), Anything(is_nullable=True)),
            (Column("a", [True, False, None]), Anything(is_nullable=True)),
            (Column("a", ["a", "b", None]), Anything(is_nullable=True)),

        ],
        ids=repr,
    )
    def test_should_return_the_data_type(self, column: Column, expected: ColumnType) -> None:
        assert ColumnType._data_type(column) == expected


class TestRepr:
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
    def test_should_create_a_printable_representation(self, column_type: ColumnType, expected: str) -> None:
        assert repr(column_type) == expected


class TestIsNullable:
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
    def test_should_return_whether_the_column_type_is_nullable(self, column_type: ColumnType, expected: bool) -> None:
        assert column_type.is_nullable() == expected


class TestIsNumeric:
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
    def test_should_return_whether_the_column_type_is_numeric(self, column_type: ColumnType, expected: bool) -> None:
        assert column_type.is_numeric() == expected

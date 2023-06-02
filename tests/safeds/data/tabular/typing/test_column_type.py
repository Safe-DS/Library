import pandas as pd
import pytest

from safeds.data.tabular.typing import (
    Anything,
    Boolean,
    ColumnType,
    Integer,
    RealNumber,
    String,
    Nothing,
)


class TestDataType:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (pd.Series([1, 2, 3]), Integer(is_nullable=False)),
            (pd.Series([1.0, 2.0, 3.0]), RealNumber(is_nullable=False)),
            (pd.Series([True, False, True]), Boolean(is_nullable=False)),
            (pd.Series(["a", "b", "c"]), String(is_nullable=False)),
            (pd.Series(["a", 1, 2.0]), Anything(is_nullable=False)),
            (pd.Series([None, None, None]), Nothing()),
            (pd.Series([1, 2, None]), Integer(is_nullable=True)),
            (pd.Series([1.0, 2.0, None]), RealNumber(is_nullable=True)),
            (pd.Series([True, False, None]), Boolean(is_nullable=True)),
            (pd.Series(["a", None, "b"]), String(is_nullable=True)),

        ],
        ids=repr,
    )
    def test_should_return_the_data_type(self, data: pd.Series, expected: ColumnType) -> None:
        assert ColumnType._data_type(data) == expected


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

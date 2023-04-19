from typing import Any

import polars as pl
import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.exceptions import UnknownColumnNameError
from safeds.data.tabular.typing import ColumnType, Integer, Schema, String


class TestFromDict:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                {},
                Row(pl.DataFrame()),
            ),
            (
                {
                    "a": 1,
                    "b": 2,
                },
                Row(pl.DataFrame({"a": 1, "b": 2})),
            ),
        ],
    )
    def test_should_create_row_from_dict(self, data: dict[str, Any], expected: Row) -> None:
        assert Row.from_dict(data) == expected


class TestInit:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (
                Row(pl.DataFrame(), Schema({})),
                Schema({}),
            ),
            (
                Row(pl.DataFrame({"col1": 0}), Schema({"col1": Integer()})),
                Schema({"col1": Integer()}),
            ),
            (
                Row(pl.DataFrame({"col1": 0, "col2": "a"}), Schema({"col1": Integer(), "col2": String()})),
                Schema({"col1": Integer(), "col2": String()}),
            ),
        ],
    )
    def test_should_use_the_schema_if_passed(self, row: Row, expected: Schema) -> None:
        assert row._schema == expected

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row(pl.DataFrame()), Schema({})),
            (Row(pl.DataFrame({"col1": 0})), Schema({"col1": Integer()})),
        ],
    )
    def test_should_infer_the_schema_if_not_passed(self, row: Row, expected: Schema) -> None:
        assert row._schema == expected


class TestEq:
    @pytest.mark.parametrize(
        ("row1", "row2", "expected"),
        [
            (Row.from_dict({}), Row.from_dict({}), True),
            (Row.from_dict({"col1": 0}), Row.from_dict({"col1": 0}), True),
            (Row.from_dict({"col1": 0}), Row.from_dict({"col1": 1}), False),
            (Row.from_dict({"col1": 0}), Row.from_dict({"col2": 0}), False),
            (Row.from_dict({"col1": 0}), Row.from_dict({"col1": "a"}), False),
        ],
        ids=[
            "empty rows",
            "equal rows",
            "different values",
            "different columns",
            "different types",
        ],
    )
    def test_should_return_whether_two_rows_are_equal(self, row1: Row, row2: Row, expected: bool) -> None:
        assert (row1.__eq__(row2)) == expected

    @pytest.mark.parametrize(
        ("row", "other"),
        [
            (Row.from_dict({"col1": 0}), None),
            (Row.from_dict({"col1": 0}), Table([])),
        ],
        ids=[
            "Row vs. None",
            "Row vs. Table",
        ],
    )
    def test_should_return_not_implemented_if_other_is_not_row(self, row: Row, other: Any) -> None:
        assert (row.__eq__(other)) is NotImplemented


class TestGetitem:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row.from_dict({"col1": 0}), "col1", 0),
            (Row.from_dict({"col1": 0, "col2": "a"}), "col2", "a"),
        ],
    )
    def test_should_return_the_value_in_the_column(self, row: Row, column_name: str, expected: Any) -> None:
        assert row[column_name] == expected

    @pytest.mark.parametrize(
        ("row", "column_name"),
        [
            (Row.from_dict({}), "col1"),
            (Row.from_dict({"col1": 0}), "col2"),
        ],
    )
    def test_should_raise_if_column_does_not_exist(self, row: Row, column_name: str) -> None:
        with pytest.raises(UnknownColumnNameError):
            # noinspection PyStatementEffect
            row[column_name]


class TestIter:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row.from_dict({}), []),
            (Row.from_dict({"col1": 0}), ["col1"]),
        ],
    )
    def test_should_return_an_iterator_for_the_column_names(self, row: Row, expected: list[str]) -> None:
        assert list(row) == expected


class TestLen:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row.from_dict({}), 0),
            (Row.from_dict({"col1": 0, "col2": "a"}), 2),
        ],
    )
    def test_should_return_the_number_of_columns(self, row: Row, expected: int) -> None:
        assert len(row) == expected


class TestStr:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row.from_dict({}), "{}"),
            (Row.from_dict({"col1": 0}), "{'col1': 0}"),
            (Row.from_dict({"col1": 0, "col2": "a"}), "{\n    'col1': 0,\n    'col2': 'a'\n}"),
        ],
        ids=[
            "empty",
            "single column",
            "multiple columns",
        ],
    )
    def test_should_return_a_string_representation(self, row: Row, expected: str) -> None:
        assert str(row) == expected


class TestRepr:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row.from_dict({}), "Row({})"),
            (Row.from_dict({"col1": 0}), "Row({'col1': 0})"),
            (Row.from_dict({"col1": 0, "col2": "a"}), "Row({\n    'col1': 0,\n    'col2': 'a'\n})"),
        ],
        ids=[
            "empty",
            "single column",
            "multiple columns",
        ],
    )
    def test_should_return_a_string_representation(self, row: Row, expected: str) -> None:
        assert repr(row) == expected


class TestGetValue:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row.from_dict({"col1": 0}), "col1", 0),
            (Row.from_dict({"col1": 0, "col2": "a"}), "col2", "a"),
        ],
    )
    def test_should_return_the_value_in_the_column(self, row: Row, column_name: str, expected: Any) -> None:
        assert row.get_value(column_name) == expected

    @pytest.mark.parametrize(
        ("row", "column_name"),
        [
            (Row.from_dict({}), "col1"),
            (Row.from_dict({"col1": 0}), "col2"),
        ],
    )
    def test_should_raise_if_column_does_not_exist(self, row: Row, column_name: str) -> None:
        with pytest.raises(UnknownColumnNameError):
            row.get_value(column_name)


class TestHasColumn:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row.from_dict({}), "col1", False),
            (Row.from_dict({"col1": 0}), "col1", True),
            (Row.from_dict({"col1": 0}), "col2", False),
        ],
    )
    def test_should_return_whether_the_row_has_the_column(self, row: Row, column_name: str, expected: bool) -> None:
        assert row.has_column(column_name) == expected


class TestGetColumnNames:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row.from_dict({}), []),
            (Row.from_dict({"col1": 0}), ["col1"]),
        ],
    )
    def test_should_return_the_column_names(self, row: Row, expected: list[str]) -> None:
        assert row.get_column_names() == expected


class TestGetTypeOfColumn:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row.from_dict({"col1": 0}), "col1", Integer()),
            (Row.from_dict({"col1": 0, "col2": "a"}), "col2", String()),
        ],
    )
    def test_should_return_the_type_of_the_column(self, row: Row, column_name: str, expected: ColumnType) -> None:
        assert row.get_type_of_column(column_name) == expected

    @pytest.mark.parametrize(
        ("row", "column_name"),
        [
            (Row.from_dict({}), "col1"),
            (Row.from_dict({"col1": 0}), "col2"),
        ],
    )
    def test_should_raise_if_column_does_not_exist(self, row: Row, column_name: str) -> None:
        with pytest.raises(UnknownColumnNameError):
            row.get_type_of_column(column_name)


class TestCount:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row.from_dict({}), 0),
            (Row.from_dict({"col1": 0, "col2": "a"}), 2),
        ],
    )
    def test_should_return_the_number_of_columns(self, row: Row, expected: int) -> None:
        assert row.count() == expected


class TestToDict:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (
                Row(pl.DataFrame({})),
                {},
            ),
            (
                Row(pl.DataFrame({"a": 1, "b": 2})),
                {
                    "a": 1,
                    "b": 2,
                },
            ),
        ],
    )
    def test_should_return_dict_for_table(self, row: Row, expected: dict[str, Any]) -> None:
        assert row.to_dict() == expected

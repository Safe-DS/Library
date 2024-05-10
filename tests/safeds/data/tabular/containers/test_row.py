import re
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.typing import ColumnType, Integer, Schema, String
from safeds.exceptions import UnknownColumnNameError


class TestFromDict:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                {},
                Row({}),
            ),
            (
                {
                    "a": 1,
                    "b": 2,
                },
                Row({"a": 1, "b": 2}),
            ),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_create_row_from_dict(self, data: dict[str, Any], expected: Row) -> None:
        assert Row.from_dict(data) == expected


class TestFromPandasDataFrame:
    @pytest.mark.parametrize(
        ("dataframe", "schema", "expected"),
        [
            (
                pd.DataFrame({"col1": [0]}),
                Schema({"col1": String()}),
                Schema({"col1": String()}),
            ),
            (
                pd.DataFrame({"col1": [0], "col2": ["a"]}),
                Schema({"col1": String(), "col2": String()}),
                Schema({"col1": String(), "col2": String()}),
            ),
        ],
        ids=[
            "one column",
            "two columns",
        ],
    )
    def test_should_use_the_schema_if_passed(self, dataframe: pd.DataFrame, schema: Schema, expected: Schema) -> None:
        row = Row._from_pandas_dataframe(dataframe, schema)
        assert row._schema == expected

    @pytest.mark.parametrize(
        ("dataframe", "expected"),
        [
            (
                pd.DataFrame({"col1": [0]}),
                Schema({"col1": Integer()}),
            ),
            (
                pd.DataFrame({"col1": [0], "col2": ["a"]}),
                Schema({"col1": Integer(), "col2": String()}),
            ),
        ],
        ids=[
            "one column",
            "two columns",
        ],
    )
    def test_should_infer_the_schema_if_not_passed(self, dataframe: pd.DataFrame, expected: Schema) -> None:
        row = Row._from_pandas_dataframe(dataframe)
        assert row._schema == expected

    @pytest.mark.parametrize(
        "dataframe",
        [
            pd.DataFrame(),
            pd.DataFrame({"col1": [0, 1]}),
        ],
        ids=[
            "empty",
            "two rows",
        ],
    )
    def test_should_raise_if_dataframe_does_not_contain_exactly_one_row(self, dataframe: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=re.escape("The dataframe has to contain exactly one row.")):
            Row._from_pandas_dataframe(dataframe)


class TestInit:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row(), Schema({})),
            (Row({}), Schema({})),
            (Row({"col1": 0}), Schema({"col1": Integer()})),
        ],
        ids=[
            "empty",
            "empty (explicit)",
            "one column",
        ],
    )
    def test_should_infer_the_schema(self, row: Row, expected: Schema) -> None:
        assert row._schema == expected


class TestContains:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row({}), "col1", False),
            (Row({"col1": 0}), "col1", True),
            (Row({"col1": 0}), "col2", False),
            (Row({"col1": 0}), 1, False),
        ],
        ids=[
            "empty row",
            "column exists",
            "column does not exist",
            "not a string",
        ],
    )
    def test_should_return_whether_the_row_has_the_column(self, row: Row, column_name: str, expected: bool) -> None:
        assert (column_name in row) == expected


class TestEq:
    @pytest.mark.parametrize(
        ("row1", "row2", "expected"),
        [
            (Row(), Row(), True),
            (Row({"col1": 0}), Row({"col1": 0}), True),
            (Row({"col1": 0}), Row({"col1": 1}), False),
            (Row({"col1": 0}), Row({"col2": 0}), False),
            (Row({"col1": 0}), Row({"col1": "a"}), False),
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
        "row",
        [
            Row(),
            Row({"col1": 0}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_return_true_if_objects_are_identical(self, row: Row) -> None:
        assert (row.__eq__(row)) is True

    @pytest.mark.parametrize(
        ("row", "other"),
        [
            (Row({"col1": 0}), None),
            (Row({"col1": 0}), Table()),
        ],
        ids=[
            "Row vs. None",
            "Row vs. Table",
        ],
    )
    def test_should_return_not_implemented_if_other_is_not_row(self, row: Row, other: Any) -> None:
        assert (row.__eq__(other)) is NotImplemented


class TestHash:
    @pytest.mark.parametrize(
        ("row1", "row2"),
        [
            (Row(), Row()),
            (Row({"col1": 0}), Row({"col1": 0})),
        ],
        ids=[
            "empty rows",
            "equal rows",
        ],
    )
    def test_should_return_same_hash_for_equal_rows(self, row1: Row, row2: Row) -> None:
        assert hash(row1) == hash(row2)

    @pytest.mark.parametrize(
        ("row1", "row2"),
        [
            (Row({"col1": 0}), Row({"col1": 1})),
            (Row({"col1": 0}), Row({"col2": 0})),
            (Row({"col1": 0}), Row({"col1": "a"})),
        ],
        ids=[
            "different values",
            "different columns",
            "different types",
        ],
    )
    def test_should_return_different_hash_for_unequal_rows(self, row1: Row, row2: Row) -> None:
        assert hash(row1) != hash(row2)


class TestGetitem:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row({"col1": 0}), "col1", 0),
            (Row({"col1": 0, "col2": "a"}), "col2", "a"),
        ],
        ids=[
            "one column",
            "two columns",
        ],
    )
    def test_should_return_the_value_in_the_column(self, row: Row, column_name: str, expected: Any) -> None:
        assert row[column_name] == expected

    @pytest.mark.parametrize(
        ("row", "column_name"),
        [
            (Row(), "col1"),
            (Row({"col1": 0}), "col2"),
        ],
        ids=[
            "empty row",
            "column does not exist",
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
            (Row(), []),
            (Row({"col1": 0}), ["col1"]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_return_an_iterator_for_the_column_names(self, row: Row, expected: list[str]) -> None:
        assert list(row) == expected


class TestLen:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row(), 0),
            (Row({"col1": 0, "col2": "a"}), 2),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_return_the_number_of_columns(self, row: Row, expected: int) -> None:
        assert len(row) == expected


class TestRepr:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row(), "Row({})"),
            (Row({"col1": 0}), "Row({'col1': 0})"),
            (Row({"col1": 0, "col2": "a"}), "Row({\n    'col1': 0,\n    'col2': 'a'\n})"),
        ],
        ids=[
            "empty",
            "single column",
            "multiple columns",
        ],
    )
    def test_should_return_a_string_representation(self, row: Row, expected: str) -> None:
        assert repr(row) == expected


class TestStr:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row(), "{}"),
            (Row({"col1": 0}), "{'col1': 0}"),
            (Row({"col1": 0, "col2": "a"}), "{\n    'col1': 0,\n    'col2': 'a'\n}"),
        ],
        ids=[
            "empty",
            "single column",
            "multiple columns",
        ],
    )
    def test_should_return_a_string_representation(self, row: Row, expected: str) -> None:
        assert str(row) == expected


class TestColumnNames:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row(), []),
            (Row({"col1": 0}), ["col1"]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_return_the_column_names(self, row: Row, expected: list[str]) -> None:
        assert row.column_names == expected


class TestNumberOfColumns:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (Row(), 0),
            (Row({"col1": 0, "col2": "a"}), 2),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_return_the_number_of_columns(self, row: Row, expected: int) -> None:
        assert row.number_of_columns == expected


class TestGetValue:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row({"col1": 0}), "col1", 0),
            (Row({"col1": 0, "col2": "a"}), "col2", "a"),
        ],
        ids=[
            "one column",
            "two columns",
        ],
    )
    def test_should_return_the_value_in_the_column(self, row: Row, column_name: str, expected: Any) -> None:
        assert row.get_value(column_name) == expected

    @pytest.mark.parametrize(
        ("row", "column_name"),
        [
            (Row({}), "col1"),
            (Row({"col1": 0}), "col2"),
        ],
        ids=[
            "empty row",
            "column does not exist",
        ],
    )
    def test_should_raise_if_column_does_not_exist(self, row: Row, column_name: str) -> None:
        with pytest.raises(UnknownColumnNameError):
            row.get_value(column_name)


class TestHasColumn:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row(), "col1", False),
            (Row({"col1": 0}), "col1", True),
            (Row({"col1": 0}), "col2", False),
        ],
        ids=[
            "empty row",
            "column exists",
            "column does not exist",
        ],
    )
    def test_should_return_whether_the_row_has_the_column(self, row: Row, column_name: str, expected: bool) -> None:
        assert row.has_column(column_name) == expected


class TestGetColumnType:
    @pytest.mark.parametrize(
        ("row", "column_name", "expected"),
        [
            (Row({"col1": 0}), "col1", Integer()),
            (Row({"col1": 0, "col2": "a"}), "col2", String()),
        ],
        ids=[
            "one column",
            "two columns",
        ],
    )
    def test_should_return_the_type_of_the_column(self, row: Row, column_name: str, expected: ColumnType) -> None:
        assert row.get_column_type(column_name) == expected

    @pytest.mark.parametrize(
        ("row", "column_name"),
        [
            (Row(), "col1"),
            (Row({"col1": 0}), "col2"),
        ],
        ids=[
            "empty row",
            "column does not exist",
        ],
    )
    def test_should_raise_if_column_does_not_exist(self, row: Row, column_name: str) -> None:
        with pytest.raises(UnknownColumnNameError):
            row.get_column_type(column_name)


class TestToDict:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (
                Row(),
                {},
            ),
            (
                Row({"a": 1, "b": 2}),
                {
                    "a": 1,
                    "b": 2,
                },
            ),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_return_dict_for_table(self, row: Row, expected: dict[str, Any]) -> None:
        assert row.to_dict() == expected


class TestToHtml:
    @pytest.mark.parametrize(
        "row",
        [
            Row(),
            Row({"a": 1, "b": 2}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_table_element(self, row: Row) -> None:
        pattern = r"<table.*?>.*?</table>"
        assert re.search(pattern, row.to_html(), flags=re.S) is not None

    @pytest.mark.parametrize(
        "row",
        [
            Row(),
            Row({"a": 1, "b": 2}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_th_element_for_each_column_name(self, row: Row) -> None:
        for column_name in row.column_names:
            assert f"<th>{column_name}</th>" in row.to_html()

    @pytest.mark.parametrize(
        "row",
        [
            Row(),
            Row({"a": 1, "b": 2}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_td_element_for_each_value(self, row: Row) -> None:
        for value in row.values():
            assert f"<td>{value}</td>" in row.to_html()


class TestReprHtml:
    @pytest.mark.parametrize(
        "row",
        [
            Row(),
            Row({"a": 1, "b": 2}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_table_element(self, row: Row) -> None:
        pattern = r"<table.*?>.*?</table>"
        assert re.search(pattern, row._repr_html_(), flags=re.S) is not None

    @pytest.mark.parametrize(
        "row",
        [
            Row(),
            Row({"a": 1, "b": 2}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_th_element_for_each_column_name(self, row: Row) -> None:
        for column_name in row.column_names:
            assert f"<th>{column_name}</th>" in row._repr_html_()

    @pytest.mark.parametrize(
        "row",
        [
            Row(),
            Row({"a": 1, "b": 2}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_td_element_for_each_value(self, row: Row) -> None:
        for value in row.values():
            assert f"<td>{value}</td>" in row._repr_html_()


class TestSizeof:
    @pytest.mark.parametrize(
        "row",
        [
            Row(),
            Row({"col1": 0}),
            Row({"col1": 0, "col2": "a"}),
        ],
        ids=[
            "empty",
            "single column",
            "multiple columns",
        ],
    )
    def test_should_size_be_greater_than_normal_object(self, row: Row) -> None:
        assert sys.getsizeof(row) > sys.getsizeof(object())

from typing import Any

import pandas as pd
import pytest
import regex as re

from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import IndexOutOfBoundsError
from safeds.data.tabular.typing import ColumnType, Integer, RealNumber, String, Boolean


class TestFromPandasSeries:
    @pytest.mark.parametrize(
        ("series", "expected"),
        [
            (pd.Series([]), []),
            (pd.Series([True, False, True]), [True, False, True]),
            (pd.Series([1, 2, 3]), [1, 2, 3]),
            (pd.Series([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]),
            (pd.Series(["a", "b", "c"]), ["a", "b", "c"]),
            (pd.Series([1, 2.0, "a", True]), [1, 2.0, "a", True]),
        ],
        ids=[
            "empty",
            "boolean",
            "integer",
            "real number",
            "string",
            "mixed"
        ],
    )
    def test_should_store_the_data(self, series: pd.Series, expected: Column) -> None:
        assert list(Column._from_pandas_series(series)) == expected

    @pytest.mark.parametrize(
        ("series", "type_"),
        [
            (pd.Series([True, False, True]), Boolean()),
            (pd.Series([1, 2, 3]), Boolean()),
        ],
        ids=[
            "type is correct",
            "type is wrong"
        ],
    )
    def test_should_use_type_if_passed(self, series: pd.Series, type_: ColumnType) -> None:
        assert Column._from_pandas_series(series, type_).type == type_

    @pytest.mark.parametrize(
        ("series", "expected"),
        [
            (pd.Series([]), String()),
            (pd.Series([True, False, True]), Boolean()),
            (pd.Series([1, 2, 3]), Integer()),
            (pd.Series([1.0, 2.0, 3.0]), RealNumber()),
            (pd.Series(["a", "b", "c"]), String()),
            (pd.Series([1, 2.0, "a", True]), String()),
        ],
        ids=[
            "empty",
            "boolean",
            "integer",
            "real number",
            "string",
            "mixed"
        ],
    )
    def test_should_infer_type_if_not_passed(self, series: pd.Series, expected: ColumnType) -> None:
        assert Column._from_pandas_series(series).type == expected


class TestGetItem:
    @pytest.mark.parametrize(
        ("column", "index", "expected"),
        [
            (Column("a", [0, 1]), 0, 0),
            (Column("a", [0, 1]), 1, 1),
        ],
        ids=[
            "first item",
            "second item"
        ],
    )
    def test_should_get_the_item_at_index(self, column: Column, index: int, expected: Any) -> None:
        assert column[index] == expected

    @pytest.mark.parametrize(
        ("column", "index", "expected"),
        [
            (Column("a", [0, 1, 2]), slice(0, 1), Column("a", [0])),
            (Column("a", [0, 1, 2]), slice(2, 3), Column("a", [2])),
            (Column("a", [0, 1, 2]), slice(0, 3), Column("a", [0, 1, 2])),
            (Column("a", [0, 1, 2]), slice(0, 3, 2), Column("a", [0, 2])),
        ],
        ids=[
            "first item",
            "last item",
            "all items",
            "every other item"
        ],
    )
    def test_should_get_the_items_at_slice(self, column: Column, index: slice, expected: Column) -> None:
        assert column[index] == expected

    @pytest.mark.parametrize(
        "index",
        [
            -1,
            2,
            slice(-1, 2),
            slice(0, 4),
            slice(-1, 4)
        ],
        ids=[
            "negative",
            "out of bounds",
            "slice with negative start",
            "slice with out of bounds end",
            "slice with negative start and out of bounds end"
        ],
    )
    def test_should_raise_if_index_is_out_of_bounds(self, index: int | slice) -> None:
        column = Column("a", [0, "1"])

        with pytest.raises(IndexOutOfBoundsError):
            # noinspection PyStatementEffect
            column[index]


class TestContains:
    @pytest.mark.parametrize(
        ("column", "value", "expected"),
        [
            (Column("a", []), 1, False),
            (Column("a", [1, 2, 3]), 1, True),
            (Column("a", [1, 2, 3]), 4, False),
        ],
        ids=[
            "empty",
            "value exists",
            "value does not exist",
        ],
    )
    def test_should_check_whether_the_value_exists(self, column: Column, value: Any, expected: bool) -> None:
        assert (value in column) == expected


class TestToHtml:
    @pytest.mark.parametrize(
        "column",
        [
            Column("a", []),
            Column("a", [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_table_element(self, column: Column) -> None:
        pattern = r"<table.*?>.*?</table>"
        assert re.search(pattern, column.to_html(), flags=re.S) is not None

    @pytest.mark.parametrize(
        "column",
        [
            Column("a", []),
            Column("a", [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_th_element_for_column_name(self, column: Column) -> None:
        assert f"<th>{column.name}</th>" in column.to_html()

    @pytest.mark.parametrize(
        "column",
        [
            Column("a", []),
            Column("a", [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_td_element_for_each_value(self, column: Column) -> None:
        for value in column:
            assert f"<td>{value}</td>" in column.to_html()


class TestReprHtml:
    @pytest.mark.parametrize(
        "column",
        [
            Column("a", []),
            Column("a", [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_table_element(self, column: Column) -> None:
        pattern = r"<table.*?>.*?</table>"
        assert re.search(pattern, column._repr_html_(), flags=re.S) is not None

    @pytest.mark.parametrize(
        "column",
        [
            Column("a", []),
            Column("a", [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_th_element_for_column_name(self, column: Column) -> None:
        assert f"<th>{column.name}</th>" in column._repr_html_()

    @pytest.mark.parametrize(
        "column",
        [
            Column("a", []),
            Column("a", [1, 2, 3]),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_td_element_for_each_value(self, column: Column) -> None:
        for value in column:
            assert f"<td>{value}</td>" in column._repr_html_()

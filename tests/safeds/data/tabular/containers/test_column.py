from typing import Any

import pandas as pd
import pytest
import regex as re

from safeds.data.tabular.containers import Column
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

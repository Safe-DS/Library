from typing import Any

import pytest
import regex as re
from safeds.data.tabular.containers import Column


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

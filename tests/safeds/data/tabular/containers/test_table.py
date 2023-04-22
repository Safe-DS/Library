import re
from typing import Any

import pytest
from pandas.core.interchange.from_dataframe import from_dataframe
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import ColumnLengthMismatchError
from safeds.data.tabular.typing import Integer, Schema


class TestFromDict:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                {},
                Table([]),
            ),
            (
                {
                    "a": [1],
                    "b": [2],
                },
                Table([[1, 2]], schema=Schema({"a": Integer(), "b": Integer()})),
            ),
        ],
    )
    def test_should_create_table_from_dict(self, data: dict[str, list[Any]], expected: Table) -> None:
        assert Table.from_dict(data) == expected

    def test_should_raise_if_columns_have_different_lengths(self) -> None:
        with pytest.raises(ColumnLengthMismatchError):
            Table.from_dict({"a": [1, 2], "b": [3]})


class TestToDict:
    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            (
                Table([]),
                {},
            ),
            (
                Table([[1, 2]], schema=Schema({"a": Integer(), "b": Integer()})),
                {
                    "a": [1],
                    "b": [2],
                },
            ),
        ],
    )
    def test_should_return_dict_for_table(self, table: Table, expected: dict[str, list[Any]]) -> None:
        assert table.to_dict() == expected


class TestReprHtml:
    @pytest.mark.parametrize(
        "table",
        [
            Table.from_dict({}),
            Table.from_dict({"a": [1, 2], "b": [3, 4]}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_table_element(self, table: Table) -> None:
        pattern = r"<table.*?>.*?</table>"
        assert re.search(pattern, table._repr_html_(), flags=re.S) is not None

    @pytest.mark.parametrize(
        "table",
        [
            Table.from_dict({}),
            Table.from_dict({"a": [1, 2], "b": [3, 4]}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_th_element_for_each_column_name(self, table: Table) -> None:
        for column_name in table.column_names:
            assert f"<th>{column_name}</th>" in table._repr_html_()

    @pytest.mark.parametrize(
        "table",
        [
            Table.from_dict({}),
            Table.from_dict({"a": [1, 2], "b": [3, 4]}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_should_contain_td_element_for_each_value(self, table: Table) -> None:
        for column in table.to_columns():
            for value in column:
                assert f"<td>{value}</td>" in table._repr_html_()


class TestDataframe:
    @pytest.mark.parametrize(
        "table",
        [
            Table([]),
            Table.from_dict({"a": [1, 2], "b": [3, 4]}),
        ],
        ids=[
            "empty",
            "non-empty",
        ],
    )
    def test_can_restore_table_from_exchange_object(self, table: Table) -> None:
        exchange_object = table.__dataframe__()
        restored = Table(from_dataframe(exchange_object))

        assert restored == table

    def test_should_raise_if_allow_copy_is_false(self) -> None:
        table = Table.from_dict({})
        with pytest.raises(NotImplementedError, match="`allow_copy` must be True"):
            table.__dataframe__(allow_copy=False)

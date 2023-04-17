from typing import Any

import pytest
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
    def test_should_create_table_from_dict(self, data: dict[str, Any], expected: Table) -> None:
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
    def test_should_return_dict_for_table(self, table: Table, expected: dict[str, Any]) -> None:
        assert table.to_dict() == expected

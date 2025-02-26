import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "substring", "expected"),
    [
        ("", "", 0),
        ("", "c", None),
        ("abc", "", 0),
        ("abc", "c", 2),
        ("abc", "abc", 0),
        ("abc", "d", None),
        (None, "", None),
        ("abc", None, None),
        (None, None, None),
    ],
    ids=[
        "empty string, empty substring",
        "empty string, non-empty substring",
        "non-empty string, empty substring",
        "correct substring",
        "substring equal to string",
        "incorrect substring",
        "None as string",
        "None as substring",
        "None for both",
    ],
)
class TestShouldGetIndexOfSubstring:
    def test_plain_arguments(self, value: str | None, substring: str | None, expected: int | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.index_of(substring),
            expected,
            type_if_none=ColumnType.string(),
        )

    def test_arguments_wrapped_in_cell(self, value: str | None, substring: str | None, expected: int | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.index_of(
                Cell.constant(substring),
            ),
            expected,
            type_if_none=ColumnType.string(),
        )

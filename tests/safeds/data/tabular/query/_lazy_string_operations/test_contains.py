import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "substring", "expected"),
    [
        ("", "", True),
        ("", "c", False),
        ("abc", "", True),
        ("abc", "c", True),
        ("abc", "abc", True),
        ("abc", "d", False),
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
class TestShouldCheckIfStringContainsSubstring:
    def test_plain_arguments(self, value: str | None, substring: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.contains(substring),
            expected,
            type_if_none=ColumnType.string(),
        )

    def test_arguments_wrapped_in_cell(self, value: str | None, substring: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.contains(
                Cell.constant(substring),
            ),
            expected,
            type_if_none=ColumnType.string(),
        )

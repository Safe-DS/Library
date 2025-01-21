import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "prefix", "expected"),
    [
        ("", "", True),
        ("", "a", False),
        ("abc", "", True),
        ("abc", "a", True),
        ("abc", "abc", True),
        ("abc", "d", False),
        (None, "", None),
        ("abc", None, None),
        (None, None, None),
    ],
    ids=[
        "empty string, empty prefix",
        "empty string, non-empty prefix",
        "non-empty string, empty prefix",
        "correct prefix",
        "prefix equal to string",
        "incorrect prefix",
        "None as string",
        "None as prefix",
        "None for both",
    ],
)
class TestShouldCheckIfStringStartsWithPrefix:
    def test_plain_arguments(self, value: str | None, prefix: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.starts_with(prefix),
            expected,
            type_if_none=ColumnType.string(),
        )

    def test_arguments_wrapped_in_cell(self, value: str | None, prefix: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.starts_with(
                Cell.constant(prefix),
            ),
            expected,
            type_if_none=ColumnType.string(),
        )

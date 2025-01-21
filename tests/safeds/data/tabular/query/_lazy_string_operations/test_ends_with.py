import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "suffix", "expected"),
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
        "empty string, empty suffix",
        "empty string, non-empty suffix",
        "non-empty string, empty suffix",
        "correct suffix",
        "suffix equal to string",
        "incorrect suffix",
        "None as string",
        "None as suffix",
        "None for both",
    ],
)
class TestShouldCheckIfStringEndsWithSuffix:
    def test_plain_arguments(self, value: str | None, suffix: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.ends_with(suffix),
            expected,
            type_if_none=ColumnType.string(),
        )

    def test_arguments_wrapped_in_cell(self, value: str | None, suffix: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.ends_with(
                Cell.constant(suffix),
            ),
            expected,
            type_if_none=ColumnType.string(),
        )

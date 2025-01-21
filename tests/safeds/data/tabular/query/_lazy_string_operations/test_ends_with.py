import pytest

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
def test_should_check_if_string_ends_with_suffix(value: str | None, suffix: str | None, expected: bool | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.ends_with(suffix),
        expected,
        type_if_none=ColumnType.string(),
    )

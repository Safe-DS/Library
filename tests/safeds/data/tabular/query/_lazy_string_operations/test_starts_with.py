import pytest

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
def test_should_check_if_string_starts_with_prefix(
    value: str | None,
    prefix: str | None,
    expected: bool | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.starts_with(prefix),
        expected,
        type_if_none=ColumnType.string(),
    )

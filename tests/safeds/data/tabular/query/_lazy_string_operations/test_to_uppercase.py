import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("", ""),
        ("abc", "ABC"),
        ("ABC", "ABC"),
        ("aBc", "ABC"),
        (None, None),
    ],
    ids=[
        "empty",
        "full lowercase",
        "full uppercase",
        "mixed",
        "None",
    ],
)
def test_should_convert_string_to_uppercase(string: str | None, expected: str | None) -> None:
    assert_cell_operation_works(
        string, lambda cell: cell.str.to_uppercase(), expected, type_if_none=ColumnType.string()
    )

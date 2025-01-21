import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ""),
        ("abc", "abc"),
        ("ABC", "abc"),
        ("aBc", "abc"),
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
def test_should_convert_string_to_lowercase(value: str | None, expected: str | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_lowercase(),
        expected,
        type_if_none=ColumnType.string(),
    )

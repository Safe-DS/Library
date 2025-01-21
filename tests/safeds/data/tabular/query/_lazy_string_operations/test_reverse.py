import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ""),
        ("abc", "cba"),
        (None, None),
    ],
    ids=[
        "empty",
        "non-empty",
        "None",
    ],
)
def test_should_reverse_string(value: str | None, expected: str | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.str.reverse(), expected, type_if_none=ColumnType.string())

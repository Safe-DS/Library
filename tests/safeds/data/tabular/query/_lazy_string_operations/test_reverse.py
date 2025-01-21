import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
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
def test_should_reverse_string(string: str | None, expected: str | None) -> None:
    assert_cell_operation_works(string, lambda cell: cell.str.reverse(), expected, type_if_none=ColumnType.string())

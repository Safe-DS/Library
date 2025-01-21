import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "base", "expected"),
    [
        ("", 10, None),
        ("abc", 10, None),
        ("10", 10, 10),
        ("10", 2, 2),
        (None, 10, None),
        ("0", None, None),
        (None, None, None),
    ],
    ids=[
        "empty",
        "invalid",
        "base 10",
        "base 2",
        "None as value",
        "None as base",
        "None for both",
    ],
)
def test_should_convert_string_to_integer(value: str | None, base: int | None, expected: float | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_int(base=base),
        expected,
        type_if_none=ColumnType.string(),
    )

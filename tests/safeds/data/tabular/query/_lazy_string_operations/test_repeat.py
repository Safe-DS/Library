import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import OutOfBoundsError
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "count", "expected"),
    [
        ("", 1, ""),
        ("a", 0, ""),
        ("a", 1, "a"),
        ("a", 2, "aa"),
        (None, 0, ""),
        (None, 1, None),
        ("", None, None),
        (None, None, None),
    ],
    ids=[
        "empty",
        "zero count",
        "non-empty (count 1)",
        "non-empty (count 2)",
        "None as string (count 0)",
        "None as string (count 1)",
        "None as count",
        "None for both",
    ],
)
def test_should_repeat_string(value: str | None, count: int | None, expected: str | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.str.repeat(count), expected, type_if_none=ColumnType.string())


def test_should_raise_if_count_is_out_of_bounds() -> None:
    column = Column("a", [])
    with pytest.raises(OutOfBoundsError):
        column.transform(lambda cell: cell.str.repeat(-1))

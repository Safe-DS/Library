import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import OutOfBoundsError
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "start", "length", "expected"),
    [
        ("", 0, None, ""),
        ("abc", 0, None, "abc"),
        ("abc", 10, None, ""),
        ("abc", -1, None, "c"),
        ("abc", -10, None, "abc"),
        ("abc", 0, 1, "a"),
        ("abc", 0, 10, "abc"),
        (None, 0, 1, None),
        ("abc", None, 1, None),
        (None, None, None, None),
    ],
    ids=[
        "empty",
        "non-negative start in bounds",
        "non-negative start out of bounds",
        "negative start in bounds",
        "negative start out of bounds",
        "non-negative length in bounds",
        "non-negative length out of bounds",
        "None as string",
        "None as start",
        "None for all",
    ],
)
def test_should_slice_characters(
    value: str | None,
    start: int | None,
    length: int | None,
    expected: bool | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.slice(start=start, length=length),
        expected,
        type_if_none=ColumnType.string(),
    )


def test_should_raise_for_negative_length() -> None:
    column = Column("a", [1])
    with pytest.raises(OutOfBoundsError):
        column.transform(lambda cell: cell.str.slice(length=-1))

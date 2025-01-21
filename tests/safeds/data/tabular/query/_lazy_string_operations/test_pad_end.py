import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import OutOfBoundsError
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "length", "character", "expected"),
    [
        ("", 0, "a", ""),
        ("", 1, "a", "a"),
        ("b", 2, "a", "ba"),
        ("bc", 2, "a", "bc"),
        ("abc", 2, "a", "abc"),
        (None, 1, " ", None),
    ],
    ids=[
        "empty (length 0)",
        "empty (length 1)",
        "non-empty (shorter length)",
        "non-empty (same length)",
        "non-empty (longer length)",
        "None",
    ],
)
def test_should_pad_end(value: str | None, length: int, character: str, expected: bool | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.pad_end(length, character=character),
        expected,
        type_if_none=ColumnType.string(),
    )


def test_should_raise_if_length_is_out_of_bounds() -> None:
    column = Column("col1", [])
    with pytest.raises(OutOfBoundsError):
        column.transform(lambda cell: cell.str.pad_end(-1))


@pytest.mark.parametrize(
    "character",
    [
        "",
        "ab",
    ],
    ids=[
        "empty string",
        "multiple characters",
    ],
)
def test_should_raise_if_char_is_not_single_character(character: str) -> None:
    column = Column("col1", [])
    with pytest.raises(ValueError, match=r"Can only pad with a single character\."):
        column.transform(lambda cell: cell.str.pad_end(1, character=character))

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "characters", "expected"),
    [
        ("", " ", ""),
        ("~ a ~", "", "~ a ~"),
        ("~ a ~", "~", "~ a "),
        ("~ a ~", "~ ", "~ a"),
        (None, " ", None),
        (" \na\n ", None, " \na"),
        (None, None, None),
    ],
    ids=[
        "empty",
        "non-empty (empty characters)",
        "non-empty (one character)",
        "non-empty (multiple characters)",
        "None as string",
        "None as characters",
        "None as both",
    ],
)
def test_should_strip_end(value: str | None, characters: str | None, expected: bool | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.strip_end(characters=characters),
        expected,
        type_if_none=ColumnType.string(),
    )

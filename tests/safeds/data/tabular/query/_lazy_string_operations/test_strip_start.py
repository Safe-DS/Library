import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "characters", "expected"),
    [
        ("", " ", ""),
        ("~ a ~", "", "~ a ~"),
        ("~ a ~", "~", " a ~"),
        ("~ a ~", "~ ", "a ~"),
        (None, " ", None),
        (" \na\n ", None, "a\n "),
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
class TestShouldStripStart:
    def test_plain_arguments(self, value: str | None, characters: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.strip_start(characters=characters),
            expected,
            type_if_none=ColumnType.string(),
        )

    def test_arguments_wrapped_in_cell(self, value: str | None, characters: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.strip_start(
                characters=Cell.constant(characters),
            ),
            expected,
            type_if_none=ColumnType.string(),
        )

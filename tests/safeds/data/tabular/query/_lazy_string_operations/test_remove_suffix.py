import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "suffix", "expected"),
    [
        ("", " ", ""),
        ("~ a ~", "", "~ a ~"),
        ("~ a ~", " ~", "~ a"),
        ("~ a ~", "~ ", "~ a ~"),
        (None, " ", None),
        ("~ a ~", None, None),
        (None, None, None),
    ],
    ids=[
        "empty",
        "empty suffix",
        "non-empty (has suffix)",
        "non-empty (does not have suffix)",
        "None as string",
        "None as suffix",
        "None as both",
    ],
)
class TestShouldRemoveSuffix:
    def test_plain_arguments(self, value: str | None, suffix: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.remove_suffix(suffix),
            expected,
            type_if_none=ColumnType.string(),
        )

    def test_arguments_wrapped_in_cell(self, value: str | None, suffix: str | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.str.remove_suffix(
                Cell.constant(suffix, type=ColumnType.string()),
            ),
            expected,
            type_if_none=ColumnType.string(),
        )

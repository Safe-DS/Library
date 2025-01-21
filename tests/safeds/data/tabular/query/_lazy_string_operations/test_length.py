import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "optimize_for_ascii", "expected"),
    [
        ("", False, 0),
        ("", True, 0),
        ("abc", False, 3),
        ("abc", True, 3),
        ("a 🪲", False, 3),
        ("a 🪲", True, 6),
        (None, False, None),
    ],
    ids=[
        "empty (not optimized)",
        "empty (optimized)",
        "ASCII only (not optimized)",
        "ASCII only (optimized)",
        "unicode (not optimized)",
        "unicode (optimized)",
        "None",
    ],
)
def test_should_get_number_of_characters(value: str | None, optimize_for_ascii: bool, expected: str | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.length(optimize_for_ascii=optimize_for_ascii),
        expected,
        type_if_none=ColumnType.string(),
    )

import pytest
from helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "optimize_for_ascii", "expected"),
    [
        ("", False, 0),
        ("", True, 0),
        ("abc", False, 3),
        ("abc", True, 3),
    ],
    ids=[
        "empty (unoptimized)",
        "empty (optimized)",
        "non-empty (unoptimized)",
        "non-empty (optimized)",
    ],
)
def test_should_return_number_of_characters(string: str, optimize_for_ascii: bool, expected: bool) -> None:
    assert_cell_operation_works(
        [string],
        lambda cell: cell.string.length(optimize_for_ascii=optimize_for_ascii),
        [expected],
    )

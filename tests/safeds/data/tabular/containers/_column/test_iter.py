import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("col1", []), []),
        (Column("col1", [0]), [0]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_iterate_over_the_data(column: Column, expected: list) -> None:
    assert list(column) == expected

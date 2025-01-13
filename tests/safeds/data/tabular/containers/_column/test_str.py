import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (
            Column("col1", []),
            "+------+\n| col1 |\n| ---  |\n| null |\n+======+\n+------+",
        ),
        (
            Column("col1", [0]),
            "+------+\n| col1 |\n|  --- |\n|  i64 |\n+======+\n|    0 |\n+------+",
        ),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_a_string_representation(column: Column, expected: str) -> None:
    assert str(column) == expected

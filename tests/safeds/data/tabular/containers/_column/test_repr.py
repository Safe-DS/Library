import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (
            Column("a", []),
            "+------+\n| a    |\n| ---  |\n| null |\n+======+\n+------+",
        ),
        (
            Column("a", [0]),
            "+-----+\n|   a |\n| --- |\n| i64 |\n+=====+\n|   0 |\n+-----+",
        ),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_a_string_representation(column: Column, expected: str) -> None:
    assert repr(column) == expected

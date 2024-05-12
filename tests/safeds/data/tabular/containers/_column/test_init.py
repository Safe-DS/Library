import pytest
from safeds.data.tabular.containers import Column


def test_should_store_the_name() -> None:
    column = Column("a", [])
    assert column.name == "a"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("a"), []),
        (Column("a", None), []),
        (Column("a", []), []),
        (Column("a", [1, 2, 3]), [1, 2, 3]),
    ],
    ids=[
        "none (implicit)",
        "none (explicit)",
        "empty",
        "non-empty",
    ],
)
def test_should_store_the_data(column: Column, expected: list) -> None:
    assert list(column) == expected

from typing import Any

import pytest

from safeds.data.tabular.containers import Column


def test_should_store_the_name() -> None:
    column: Column[Any] = Column("col1", [])
    assert column.name == "col1"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("col1", []), []),
        (Column("col1", [1]), [1]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_store_the_data(column: Column, expected: list) -> None:
    assert list(column) == expected

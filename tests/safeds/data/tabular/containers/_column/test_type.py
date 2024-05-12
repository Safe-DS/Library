from typing import Any

from safeds.data.tabular.containers import Column


def test_should_return_the_type() -> None:
    column: Column[Any] = Column("a", [])
    assert str(column.type) == "Null"

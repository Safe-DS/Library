from typing import Any

import pandas as pd
import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import String, Boolean, Integer, RealNumber, ColumnType


def test_should_store_the_name() -> None:
    column: Column[Any] = Column("A", [])
    assert column.name == "A"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("A", []), "A"),
        (Column("A", pd.Series()), "A"),
    ],
    ids=[
        "data as list",
        "data as series"
    ],
)
def test_should_set_the_name_of_internal_series(column: Column, expected: str) -> None:
    assert column._data.name == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("A"), []),
        (Column("A", []), []),
        (Column("A", [1, 2, 3]), [1, 2, 3]),
    ],
    ids=[
        "empty",
        "empty (explicit)",
        "non-empty",
    ],
)
def test_should_store_the_data(column: Column, expected: list) -> None:
    assert list(column) == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("A", []), String()),
        (Column("A", [True, False, True]), Boolean()),
        (Column("A", [1, 2, 3]), Integer()),
        (Column("A", [1.0, 2.0, 3.0]), RealNumber()),
        (Column("A", ["a", "b", "c"]), String()),
        (Column("A", [1, 2.0, "a", True]), String()),
    ],
    ids=["empty", "boolean", "integer", "real number", "string", "mixed"],
)
def test_should_infer_type(column: Column, expected: ColumnType) -> None:
    assert column.type == expected

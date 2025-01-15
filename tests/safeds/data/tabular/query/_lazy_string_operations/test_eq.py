from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.query import StringOperations
from safeds.data.tabular.query._lazy_string_operations import _LazyStringOperations


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyStringOperations(pl.col("a")), _LazyStringOperations(pl.col("a")), True),
        (_LazyStringOperations(pl.col("a")), _LazyStringOperations(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_return_whether_two_cells_are_equal(
    cell1: StringOperations,
    cell2: StringOperations,
    expected: bool,
) -> None:
    assert (cell1.__eq__(cell2)) == expected


def test_should_return_true_if_objects_are_identical() -> None:
    cell = _LazyStringOperations(pl.col("a"))
    assert (cell.__eq__(cell)) is True


@pytest.mark.parametrize(
    ("cell", "other"),
    [
        (_LazyStringOperations(pl.col("a")), None),
        (_LazyStringOperations(pl.col("a")), Table({})),
    ],
    ids=[
        "Cell vs. None",
        "Cell vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_cell(cell: StringOperations, other: Any) -> None:
    assert (cell.__eq__(other)) is NotImplemented

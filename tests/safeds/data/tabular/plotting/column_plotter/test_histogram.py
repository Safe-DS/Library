import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "column",
    [
        Column("a", []),
        Column("a", [0]),
        Column("a", [0, 1]),
        Column("a", ["A", "B", "C"]),
    ],
    ids=[
        "empty",
        "one row (numeric)",
        "multiple rows (numeric)",
        "non-numeric",
    ],
)
def test_should_match_snapshot(column: Column, snapshot_png_image: SnapshotAssertion) -> None:
    histogram = column.plot.histogram()
    assert histogram == snapshot_png_image


@pytest.mark.parametrize(
    "column",
    [
        Column("a", []),
        Column("a", [0]),
        Column("a", [0, 1]),
        Column("a", ["A", "B", "C"]),
    ],
    ids=[
        "empty",
        "one row (numeric)",
        "multiple rows (numeric)",
        "non-numeric",
    ],
)
def test_should_match_snapshot_dark(column: Column, snapshot_png_image: SnapshotAssertion) -> None:
    histogram = column.plot.histogram(theme="dark")
    assert histogram == snapshot_png_image

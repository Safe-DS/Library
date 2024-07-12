import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError, OutOfBoundsError
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    ("table", "col1", "col2"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "B"),
        (
            Table(
                {
                    "A": [1, 0.99, 0.99, 2],
                    "B": [1, 0.99, 1.01, 2],
                },
            ),
            "A",
            "B",
        ),
        (
            Table(
                {"A": [1, 0.99, 0.99, 2], "B": [1, 0.99, 1.01, 2], "C": [2, 2.99, 2.01, 3]},
            ),
            "A",
            "B",
        ),
    ],
    ids=[
        "functional",
        "overlapping",
        "multiple",
    ],
)
def test_should_match_snapshot(
    table: Table,
    col1: str,
    col2: str,
    snapshot_png_image: SnapshotAssertion,
) -> None:
    histogram_2d = table.plot.histogram_2d(col1, col2)
    assert histogram_2d == snapshot_png_image


@pytest.mark.parametrize(
    ("table", "col1", "col2"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "A"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "B", "C"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "D"),
        (Table(), "C", "D"),
    ],
    ids=[
        "First argument doesn't exist",
        "Second argument doesn't exist",
        "Both arguments do not exist",
        "empty",
    ],
)
def test_should_raise_if_column_does_not_exist(table: Table, col1: str, col2: str) -> None:
    with pytest.raises(ColumnNotFoundError):
        table.plot.histogram_2d(col1, col2)


@pytest.mark.parametrize(
    ("table", "col1", "col2"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "B"),
    ],
    ids=["dark_theme"],
)
def test_should_match_snapshot_dark_theme(
    table: Table,
    col1: str,
    col2: str,
    snapshot_png_image: SnapshotAssertion,
) -> None:
    histogram_2d = table.plot.histogram_2d(col1, col2, theme="dark")
    assert histogram_2d == snapshot_png_image


def test_should_raise_if_value_not_in_range_x() -> None:
    table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(OutOfBoundsError):
        table.plot.histogram_2d("col1", "col2", x_max_bin_count=0)


def test_should_raise_if_value_not_in_range_y() -> None:
    table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(OutOfBoundsError):
        table.plot.histogram_2d("col1", "col2", y_max_bin_count=0)


def test_should_raise_if_column_is_not_numeric() -> None:
    table = Table({"col1": "a", "col2": "b"})
    with pytest.raises(ColumnTypeError):
        table.plot.histogram_2d("col1", "col2")

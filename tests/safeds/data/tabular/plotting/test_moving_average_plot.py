import datetime

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    ("table", "x_name", "y_name", "window_size"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "B", 2),
        # (Table({"A": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}), "A", "B", 2),
        (
            Table(
                {
                    "time": [
                        datetime.date(2022, 1, 10),
                        datetime.date(2022, 1, 10),
                        datetime.date(2022, 1, 11),
                        datetime.date(2022, 1, 11),
                        datetime.date(2022, 1, 12),
                        datetime.date(2022, 1, 12),
                    ],
                    "A": [10, 5, 20, 2, 1, 1],
                },
            ),
            "time",
            "A",
            2,
        ),
        (
            Table(
                {
                    "time": [
                        datetime.date(2022, 1, 9),
                        datetime.date(2022, 1, 10),
                        datetime.date(2022, 1, 11),
                        datetime.date(2022, 1, 12),
                    ],
                    "A": [10, 5, 20, 2],
                },
            ),
            "time",
            "A",
            2,
        ),
    ],
    ids=["numerical", "date grouped", "date"],
)
def test_should_match_snapshot(
    table: Table,
    x_name: str,
    y_name: str,
    window_size: int,
    snapshot_png_image: SnapshotAssertion,
) -> None:
    line_plot = table.plot.moving_average_plot(x_name, y_name, window_size)
    assert line_plot == snapshot_png_image


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ("C", "A"),
        ("A", "C"),
        ("C", "D"),
    ],
    ids=["x column", "y column", "x and y column"],
)
def test_should_raise_if_column_does_not_exist_error_message(x: str, y: str) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(ColumnNotFoundError):
        table.plot.moving_average_plot(x, y, window_size=2)


@pytest.mark.parametrize(
    ("table"),
    [
        (Table({"A": [1, 2, 3], "B": ["2", 4, 7]})),
        (Table({"A": ["1", 2, 3], "B": [2, 4, 7]})),
    ],
    ids=["x column", "y column"],
)
def test_should_raise_if_column_is_not_numerical(table: Table) -> None:
    with pytest.raises(ColumnTypeError):
        table.plot.moving_average_plot("A", "B", window_size=2)


@pytest.mark.parametrize(
    ("table", "column_name"),
    [
        (Table({"A": [1, 2, 3], "B": [None, 4, 7]}), 'B'),
        (Table({"A": [None, 2, 3], "B": [2, 4, 7]}), 'A'),
    ],
    ids=["x column", "y column"],
)

def test_should_raise_if_column_has_missing_value(table: Table, column_name: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"there are missing values in column '{column_name}', use transformation to fill missing "
        f"values or drop the missing values",
    ):
        table.plot.moving_average_plot("A", "B", window_size=2)

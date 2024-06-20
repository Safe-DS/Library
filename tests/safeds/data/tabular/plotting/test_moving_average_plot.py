import datetime

import pytest
from safeds.data.tabular.containers import Table
from syrupy import SnapshotAssertion


@pytest.mark.parametrize(
    ("table", "x_name", "y_name", "window_size"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "B", 2),
        #(Table({"A": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}), "A", "B", 2),
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

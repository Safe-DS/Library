from statistics import stdev

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": ["a", "b", "c"]}),
            Table(
                {
                    "metric": [
                        "min",
                        "max",
                        "mean",
                        "median",
                        "standard deviation",
                        "distinct value count",
                        "idness",
                        "missing value ratio",
                        "stability",
                    ],
                    "col1": [
                        1,
                        2,
                        4 / 3,
                        1,
                        stdev([1, 2, 1]),
                        2,
                        2 / 3,
                        0,
                        2 / 3,
                    ],
                    "col2": [
                        "a",
                        "c",
                        "-",
                        "-",
                        "-",
                        "3",
                        "1.0",
                        "0.0",
                        "0.3333333333333333",
                    ],
                },
            ),
        ),
        (
            Table(),
            Table(),
        ),
        (
            Table({"col": [], "gg": []}),
            Table(
                {
                    "metric": [
                        "min",
                        "max",
                        "mean",
                        "median",
                        "standard deviation",
                        "distinct value count",
                        "idness",
                        "missing value ratio",
                        "stability",
                    ],
                    "col": [
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "0",
                        "1.0",
                        "1.0",
                        "1.0",
                    ],
                    "gg": [
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "0",
                        "1.0",
                        "1.0",
                        "1.0",
                    ],
                },
            ),
        ),
        (
            Table({"col": [None, None]}),
            Table(
                {
                    "metric": [
                        "min",
                        "max",
                        "mean",
                        "median",
                        "standard deviation",
                        "distinct value count",
                        "idness",
                        "missing value ratio",
                        "stability",
                    ],
                    "col": ["-", "-", "-", "-", "-", "0", "0.5", "1.0", "1.0"],
                },
            ),
        ),
        (
            Table({"col": [True, False, True]}),
            Table(
                {
                    "metric": [
                        "min",
                        "max",
                        "mean",
                        "median",
                        "standard deviation",
                        "distinct value count",
                        "idness",
                        "missing value ratio",
                        "stability",
                    ],
                    "col": [
                        "-",
                        "True",
                        "-",
                        "-",
                        "-",
                        "2",
                        "0.6666666666666666",
                        "0.0",
                        "0.6666666666666666",
                    ],
                },
            ),

        )
    ],
    ids=[
        "Column of integers and Column of characters",
        "empty",
        "empty with columns",
        "Column of None",
        "Column of booleans",
    ],
)
def test_should_summarize_statistics(table: Table, expected: Table) -> None:
    assert table.summarize_statistics() == expected

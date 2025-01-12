import datetime
from statistics import stdev

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        # empty
        (
            Table({}),
            Table({}),
        ),
        # no rows, multiple columns
        (
            Table({"col1": [], "col2": []}),
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
                    "col2": [
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
        # null column
        (
            Table({"col1": [None, None, None]}),
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
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "0",
                        "0.3333333333333333",
                        "1.0",
                        "1.0",
                    ],
                },
            ),
        ),
        # numeric column
        (
            Table({"col1": [1, 2, 1]}),
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
                },
            ),
        ),
        # temporal column
        (
            Table(
                {
                    "col1": [
                        datetime.time(1, 2, 3),
                        datetime.time(4, 5, 6),
                        datetime.time(7, 8, 9),
                    ],
                },
            ),
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
                        "01:02:03",
                        "07:08:09",
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
        # string column
        (
            Table({"col1": ["a", "b", "c"]}),
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
    ],
    ids=[
        "empty",
        "no rows, multiple columns",
        "null column",
        "numeric column",
        "temporal column",
        "string column",
    ],
)
def test_should_summarize_statistics(table: Table, expected: Table) -> None:
    assert table.summarize_statistics() == expected

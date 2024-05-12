from statistics import stdev

import pytest
from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (
            Column("col", [1, 2, 1]),
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
                        1,
                        2,
                        4.0 / 3,
                        1.0,
                        stdev([1, 2, 1]),
                        2,
                        2.0 / 3,
                        0.0,
                        2.0 / 3,
                    ],
                },
            ),
        ),
        (
            Column("col", ["a", "b", "c"]),
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
                        "a",
                        "c",
                        "-",
                        "-",
                        "-",
                        "3",
                        "1.0",
                        "0.0",
                        str(1.0 / 3),
                    ],
                },
            ),
        ),
        (
            Column("col", [None, None]),
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
                        "0.5",
                        "1.0",
                        "1.0",
                    ],
                },
            ),
        ),
    ],
    ids=[
        "Column of ints",
        "Column of strings",
        "Column of None",
    ],
)
def test_should_summarize_statistics(column: Column, expected: Table) -> None:
    assert column.summarize_statistics().schema == expected.schema
    assert column.summarize_statistics() == expected

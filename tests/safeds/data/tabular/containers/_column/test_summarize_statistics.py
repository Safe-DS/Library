from statistics import stdev

import pytest
from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (  # boolean
            Column("col", [True, False, True]),
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
                        "False",
                        "True",
                        "-",
                        "-",
                        "-",
                        "2",
                        str(2 / 3),
                        "0.0",
                        str(2 / 3),
                    ],
                },
            ),
        ),
        (  # ints
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
        (  # strings
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
        (  # only missing
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
        "boolean",
        "ints",
        "strings",
        "only missing",
    ],
)
def test_should_summarize_statistics(column: Column, expected: Table) -> None:
    assert column.summarize_statistics().schema == expected.schema
    assert column.summarize_statistics() == expected

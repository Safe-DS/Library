from statistics import stdev

import pytest
from safeds.data.tabular.containers import Table, Column


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (
            Column("col1", [1, 2, 1]),
            Table(
                {
                    "metric": [
                        "minimum",
                        "maximum",
                        "mean",
                        "mode",
                        "median",
                        "variance",
                        "standard deviation",
                        "missing value count",
                        "missing value ratio",
                        "idness",
                        "stability",
                    ],
                    "col1": [
                        "1",
                        "2",
                        str(4.0 / 3),
                        "[1]",
                        "1.0",
                        str(1.0 / 3),
                        str(stdev([1, 2, 1])),
                        "0",
                        "0.0",
                        str(2.0 / 3),
                        str(2.0 / 3),
                    ],
                },
            ),
        ),
        (
            Column("col1", ["a", "b", "c"]),
            Table(
                {
                    "metric": [
                        "minimum",
                        "maximum",
                        "mean",
                        "mode",
                        "median",
                        "variance",
                        "standard deviation",
                        "missing value count",
                        "missing value ratio",
                        "idness",
                        "stability",
                    ],
                    "col1": [
                        "-",
                        "-",
                        "-",
                        "['a', 'b', 'c']",
                        "-",
                        "-",
                        "-",
                        "0",
                        "0.0",
                        "1.0",
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
                        "minimum",
                        "maximum",
                        "mean",
                        "mode",
                        "median",
                        "variance",
                        "standard deviation",
                        "missing value count",
                        "missing value ratio",
                        "idness",
                        "stability",
                    ],
                    "col": ["-", "-", "-", "[]", "-", "-", "-", "2", "1.0", "0.0", "-"],
                },
            ),
        ),
    ],
    ids=[
        "Column of integers", 
        "Column of characters",
        "Column of None",
    ],
)
def test_should_summarize_statistics(column: Column, expected: Table) -> None:
    assert column.summarize_statistics().schema == expected.schema
    assert column.summarize_statistics() == expected
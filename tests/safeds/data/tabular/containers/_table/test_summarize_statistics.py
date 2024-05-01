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
                    "col2": [
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
            Table(),
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
                },
            ),
        ),
        (
            Table({"col": [], "gg": []}),
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
                    "col": [
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    ],
                    "gg": [
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    ],
                },
            ),
        ),
        (
            Table({"col": [None, None]}),
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
        "Column of integers and Column of characters",
        "empty",
        "empty with columns",
        "Column of None",
    ],
)
def test_should_summarize_statistics(table: Table, expected: Table) -> None:
    assert table.summarize_statistics().schema == expected.schema
    assert table.summarize_statistics() == expected

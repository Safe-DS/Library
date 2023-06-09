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
                    "metrics": [
                        "maximum",
                        "minimum",
                        "mean",
                        "mode",
                        "median",
                        "sum",
                        "variance",
                        "standard deviation",
                        "idness",
                        "stability",
                    ],
                    "col1": [
                        "2",
                        "1",
                        str(4.0 / 3),
                        "[1]",
                        "1.0",
                        "4",
                        str(1.0 / 3),
                        str(stdev([1, 2, 1])),
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
                        "-",
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
                    "metrics": [
                        "maximum",
                        "minimum",
                        "mean",
                        "mode",
                        "median",
                        "sum",
                        "variance",
                        "standard deviation",
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
                    "metrics": [
                        "maximum",
                        "minimum",
                        "mean",
                        "mode",
                        "median",
                        "sum",
                        "variance",
                        "standard deviation",
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
                    ],
                },
            ),
        ),
    ],
    ids=["Column of integers and Column of characters", "empty", "empty with columns"],
)
def test_should_make_summary(table: Table, expected: Table) -> None:
    assert expected.schema == table.summary().schema
    assert expected == table.summary()

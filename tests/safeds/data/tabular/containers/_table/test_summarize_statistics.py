import datetime
from statistics import stdev

import pytest
from safeds.data.tabular.containers import Table

_HEADERS = [
    "min",
    "max",
    "mean",
    "median",
    "standard deviation",
    "missing value ratio",
    "stability",
    "idness",
]
_EMPTY_COLUMN_RESULT = [
    None,
    None,
    None,
    None,
    None,
    1.0,
    1.0,
    1.0,
]


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
                    "statistic": _HEADERS,
                    "col1": _EMPTY_COLUMN_RESULT,
                    "col2": _EMPTY_COLUMN_RESULT,
                },
            ),
        ),
        # null column
        (
            Table({"col1": [None, None, None]}),
            Table(
                {
                    "statistic": _HEADERS,
                    "col1": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        1.0,
                        1.0,
                        1 / 3,
                    ],
                },
            ),
        ),
        # numeric column
        (
            Table({"col1": [1, 2, 1, None]}),
            Table(
                {
                    "statistic": _HEADERS,
                    "col1": [
                        1,
                        2,
                        4 / 3,
                        1,
                        stdev([1, 2, 1]),
                        1 / 4,
                        2 / 3,
                        3 / 4,
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
                        None,
                    ],
                },
            ),
            Table(
                {
                    "statistic": _HEADERS,
                    "col1": [
                        "01:02:03",
                        "07:08:09",
                        None,
                        None,
                        None,
                        "0.25",
                        "0.3333333333333333",
                        "1.0",
                    ],
                },
            ),
        ),
        # string column
        (
            Table({"col1": ["a", "b", "c", None]}),
            Table(
                {
                    "statistic": _HEADERS,
                    "col1": [
                        "a",
                        "c",
                        None,
                        None,
                        None,
                        "0.25",
                        "0.3333333333333333",
                        "1.0",
                    ],
                },
            ),
        ),
        # boolean column
        (
            Table({"col1": [True, False, True, None]}),
            Table(
                {
                    "statistic": _HEADERS,
                    "col1": [
                        "false",
                        "true",
                        None,
                        None,
                        None,
                        "0.25",
                        "0.6666666666666666",
                        "0.75",
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
        "boolean column",
    ],
)
def test_should_summarize_statistics(table: Table, expected: Table) -> None:
    actual = table.summarize_statistics()
    assert actual == expected


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        # has statistic column
        (
            Table({"statistic": []}),
            Table(
                {
                    "statistic_": _HEADERS,
                    "statistic": _EMPTY_COLUMN_RESULT,
                },
            ),
        ),
        # has statistic_ column
        (
            Table({"statistic": [], "statistic_": []}),
            Table(
                {
                    "statistic__": _HEADERS,
                    "statistic": _EMPTY_COLUMN_RESULT,
                    "statistic_": _EMPTY_COLUMN_RESULT,
                },
            ),
        ),
    ],
    ids=[
        "has statistic column",
        "has statistic_ column",
    ],
)
def test_should_ensure_new_column_has_unique_name(table: Table, expected: Table) -> None:
    actual = table.summarize_statistics()
    assert actual == expected

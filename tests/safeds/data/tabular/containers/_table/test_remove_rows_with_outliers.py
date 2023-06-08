import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table(
                {
                    "col1": ["A", "B", "C"],
                    "col2": [1.0, 2.0, 3.0],
                    "col3": [2, 3, 1],
                },
            ),
            Table(
                {
                    "col1": ["A", "B", "C"],
                    "col2": [1.0, 2.0, 3.0],
                    "col3": [2, 3, 1],
                },
            ),
        ),
        (
            Table(
                {
                    "col1": [
                        "A",
                        "B",
                        "A",
                        "outlier",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                    ],
                    "col2": [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
                    "col3": [2, 3, 1, 1_000_000_000, 1, 1, 1, 1, 1, 1, 1, 1],
                },
            ),
            Table(
                {
                    "col1": [
                        "A",
                        "B",
                        "A",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                        "a",
                    ],
                    "col2": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
                    "col3": [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                },
            ),
        ),
        (
            Table(
                {
                    "col1": [],
                    "col2": [],
                },
            ),
            Table(
                {
                    "col1": [],
                    "col2": [],
                },
            ),
        ),
        (Table(), Table()),
    ],
    ids=["no outliers", "with outliers", "no rows", "empty"],
)
def test_should_remove_rows_with_outliers(table: Table, expected: Table) -> None:
    updated_table = table.remove_rows_with_outliers()
    assert updated_table.schema == expected.schema
    assert updated_table.number_of_rows == expected.number_of_rows
    assert updated_table == expected

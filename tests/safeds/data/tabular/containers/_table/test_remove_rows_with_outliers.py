import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [(Table.from_dict(
        {
            "col1": ["A", "B", "C"],
            "col2": [1.0, 2.0, 3.0],
            "col3": [2, 3, 1],
        },
    ), 3),
        (
            Table.from_dict(
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
            ), 11),
        (
            Table.from_dict(
                {
                    "col1": [],
                    "col2": [],
                },
            ), 0),

    ],
    ids=["no outliers", "with outliers", "no rows"],
)
def test_should_remove_rows_with_no_outliers(table: Table, expected: int) -> None:
    updated_table = table.remove_rows_with_outliers()
    assert updated_table.number_of_rows == expected

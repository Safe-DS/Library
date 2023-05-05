import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [(Table.from_dict(
        {
            "col1": [None, None, "C", "A"],
            "col3": [None, 2, 3, 4],
        },
    ), 2), (Table([]), 0),

    ],
    ids=["some missing values", "empty"],
)
def test_should_remove_rows_with_missing_values(table: Table, expected: int) -> None:
    updated_table = table.remove_rows_with_missing_values()
    assert updated_table.number_of_rows == expected


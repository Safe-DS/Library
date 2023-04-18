import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table.from_dict(
                {
                    "A": [1, 1, 1, 4],
                    "B": [2, 2, 2, 5],
                },
            ),
            Table.from_dict({"A": [1, 4], "B": [2, 5]}),
        ),
        (
            Table.from_dict(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                },
            ),
            Table.from_dict({"A": [1, 4], "B": [2, 5]}),
        ),
    ],
)
def test_remove_duplicate_rows(table: Table, expected: Table) -> None:
    result_table = table.remove_duplicate_rows()
    assert result_table == expected

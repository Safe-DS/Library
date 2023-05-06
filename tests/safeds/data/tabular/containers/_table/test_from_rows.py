import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.data.tabular.exceptions import SchemaMismatchError


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [],
            Table({}),
        ),
        (
            [
                Row({"A": 1, "B": 4, "C": "d"}),
                Row({"A": 2, "B": 5, "C": "e"}),
                Row({"A": 3, "B": 6, "C": "f"}),
            ],
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["d", "e", "f"],
                },
            ),
        ),
    ],
    ids=["empty", "non-empty"],
)
def test_should_create_table_from_rows(rows: list[Row], expected: Table) -> None:
    assert Table.from_rows(rows) == expected


def test_should_raise_error_if_mismatching_schema() -> None:
    rows = [Row({"A": 1, "B": 2}), Row({"A": 2, "B": "a"})]
    with pytest.raises(SchemaMismatchError):
        Table.from_rows(rows)

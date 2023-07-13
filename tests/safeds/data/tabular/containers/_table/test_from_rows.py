import pytest
from safeds.data.tabular.containers import Row, Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [],
            Table(),
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
        (
            [
                Row({"A": 1, "B": 4, "C": "d"}),
                Row({"A": 2, "B": 5, "C": "e"}),
                Row({"A": 3, "B": "6", "C": "f"}),
            ],
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, "6"],
                    "C": ["d", "e", "f"],
                },
            ),
        ),
    ],
    ids=["empty", "non-empty", "different schemas"],
)
def test_should_create_table_from_rows(rows: list[Row], expected: Table) -> None:
    table = Table.from_rows(rows)
    assert table.schema == expected.schema
    assert table == expected


@pytest.mark.parametrize(
    ("rows", "expected_error_msg"),
    [([Row({"A": 1, "B": 2}), Row({"A": 2, "C": 4})], r"Could not find column\(s\) 'B'")],
)
def test_should_raise_error_if_unknown_column_names(rows: list[Row], expected_error_msg: str) -> None:
    with pytest.raises(UnknownColumnNameError, match=expected_error_msg):
        Table.from_rows(rows)

import pytest
from safeds.data.tabular.containers import Row, Table, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("rows", "target_name", "feature_names"),
    [
        (
            [
                Row(
                    {"feature_a": 0, "feature_b": 3, "no_feature": 6, "target": 9}
                ),
                Row(
                    {"feature_a": 1, "feature_b": 4, "no_feature": 7, "target": 10}
                ),
                Row(
                    {"feature_a": 2, "feature_b": 5, "no_feature": 8, "target": 11}
                ),
            ],
            "target",
            ["feature_a", "feature_b"],
        ),
    ],
    ids=["table"]
)
def test_should_shuffle_rows(rows: list[Row], target_name: str, feature_names: list[str]) -> None:
    table = TaggedTable._from_table(Table.from_rows(rows), target_name=target_name, feature_names=feature_names)
    shuffled = table.shuffle_rows()
    assert table.schema == shuffled.schema
    assert table.features.column_names == shuffled.features.column_names
    assert table.target.name == shuffled.target.name

    # Check that shuffled contains the original rows:
    for i in range(table.number_of_rows):
        assert shuffled.get_row(i) in rows

    # Assert that table and shuffled are equal after sorting:
    def comparator(r1: Row, r2: Row) -> int:
        return 1 if r1.__repr__() < r2.__repr__() else -1

    assert_that_tagged_tables_are_equal(table.sort_rows(comparator), shuffled.sort_rows(comparator))

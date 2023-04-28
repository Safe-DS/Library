from safeds.data.tabular.containers import Table
import pytest

@pytest.mark.parametrize(
    ("table"),
    [
        (Table.from_dict({"col1": [1], "col2": [1]})),
    ],
    ids=["Table with identical values in rows"],
)
def test_should_shuffle_rows(table: Table) -> None:
    result_table = table.shuffle_rows()
    assert table == result_table

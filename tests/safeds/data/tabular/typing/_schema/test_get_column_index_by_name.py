from safeds.data.tabular.containers import Table


def test_get_column_index_by_name() -> None:
    table = Table.from_dict({"col1": [1], "col2": [2]})
    assert table.schema._get_column_index_by_name("col1") == 0
    assert table.schema._get_column_index_by_name("col2") == 1

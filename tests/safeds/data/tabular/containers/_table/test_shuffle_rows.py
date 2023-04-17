from safeds.data.tabular.containers import Table


def test_shuffle_rows_valid() -> None:
    table = Table.from_dict({"col1": [1], "col2": [1]})
    result_table = table.shuffle_rows()
    assert table == result_table

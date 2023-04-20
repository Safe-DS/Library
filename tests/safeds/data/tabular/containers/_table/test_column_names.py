from safeds.data.tabular.containers import Table


def test_column_names() -> None:
    table = Table.from_dict({"col1": [1], "col2": [1]})
    assert table.column_names == ["col1", "col2"]


def test_get_column_names_empty() -> None:
    table = Table([])
    assert table.column_names == []

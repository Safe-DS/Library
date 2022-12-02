from safe_ds.data import Row, Table


def test_from_rows():
    table_expected = Table.from_json("tests/resources/test_row_table.json")
    rows_is: list[Row] = table_expected.to_rows()
    table_is: Table = Table.from_rows(rows_is)

    assert table_is._data.equals(table_expected._data)

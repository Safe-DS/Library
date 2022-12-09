from safe_ds.data import Row, Table


def test_from_rows() -> None:
    table_expected = Table.from_json("tests/resources/test_row_table.json")
    rows_is: list[Row] = table_expected.to_rows()
    table_is: Table = Table.from_rows(rows_is)

    assert table_is == table_expected

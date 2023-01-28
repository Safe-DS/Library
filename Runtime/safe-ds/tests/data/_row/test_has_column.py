from safeds.data import Table


def test_has_column_positive() -> None:
    table = Table.from_csv("tests/resources/test_table_has_column.csv")
    row = table.to_rows()[0]
    assert row.has_column("A")


def test_has_column_negative() -> None:
    table = Table.from_csv("tests/resources/test_table_has_column.csv")
    row = table.to_rows()[0]
    assert not row.has_column("C")

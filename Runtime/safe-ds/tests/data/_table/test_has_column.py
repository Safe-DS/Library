from safeds.data import Table


def test_has_column_positive() -> None:
    table = Table.from_csv("tests/resources/test_table_has_column.csv")
    assert table.has_column("A")


def test_has_column_negative() -> None:
    table = Table.from_csv("tests/resources/test_table_has_column.csv")
    assert not table.has_column("C")

from safeds.data import Table


def test_has_column_true() -> None:
    table = Table.from_json("tests/resources/test_schema_table.json")

    assert table.schema.has_column("A")


def test_has_column_false() -> None:
    table = Table.from_json("tests/resources/test_schema_table.json")

    assert not table.schema.has_column("C")

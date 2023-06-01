from safeds.data.tabular.containers import TaggedTable


def assert_that_tagged_tables_are_equal(table1: TaggedTable, table2: TaggedTable) -> None:
    assert table1.schema == table2.schema
    assert table1.features == table2.features
    assert table1.target == table2.target
    assert table1 == table2

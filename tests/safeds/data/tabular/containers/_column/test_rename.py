from safeds.data.tabular.containers import Column


def test_should_return_new_column_with_new_name() -> None:
    column = Column("A", [1, 2, 3])
    new_column = column.rename("B")
    assert new_column.name == "B"
    assert new_column._data.name == "B"


def test_should_not_change_name_of_original_column() -> None:
    column = Column("A", [1, 2, 3])
    column.rename("B")
    assert column.name == "A"
    assert column._data.name == "A"

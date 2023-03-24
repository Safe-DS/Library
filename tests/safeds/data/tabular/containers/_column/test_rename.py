from safeds.data.tabular.containers import Column


def test_should_return_new_column_with_new_name():
    column = Column([1, 2, 3], "A")
    new_column = column.rename("B")
    assert new_column.name == "B"


def test_should_not_change_name_of_original_column():
    column = Column([1, 2, 3], "A")
    column.rename("B")
    assert column.name == "A"

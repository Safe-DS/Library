from safeds.data.tabular import Column


def test_count_valid() -> None:
    column = Column([1, 2, 3, 4, 5], "col1")
    assert column.count() == 5

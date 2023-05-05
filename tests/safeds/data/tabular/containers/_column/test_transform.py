from safeds.data.tabular.containers import Column


def test_should_transform_column() -> None:
    column1 = Column("test", [1, 2]).transform(lambda it: it + 1)
    column2 = Column("test", [2, 3])

    assert column1 == column2

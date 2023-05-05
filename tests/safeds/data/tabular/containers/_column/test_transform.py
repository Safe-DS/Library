from safeds.data.tabular.containers import Column


def test_should_transform_column() -> None:
    column = Column("test", [1, 2])
    column = column.transform(lambda it: it + 1)

    assert column[0] == 2
    assert column[1] == 3

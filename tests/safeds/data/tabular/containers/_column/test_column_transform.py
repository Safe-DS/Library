from safeds.data.tabular.containers import Column


def test_transform():
    column = Column("test", [1, 2])
    column.transform(lambda it: it + 1)

    print(column)
    assert column[0] == 2
    assert column[1] == 3

from safeds.data.tabular.containers import Column


def test_from_columns() -> None:
    column1 = Column("A", [1, 4])
    column2 = Column("B", [2, 5])

    assert column1._type == column2._type


def test_from_columns_negative() -> None:
    column1 = Column("A", [1, 4])
    column2 = Column("B", ["2", "5"])

    assert column1._type != column2._type

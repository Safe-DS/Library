from safeds.data.tabular.containers import Column


def test_should_return_the_name() -> None:
    column = Column("col1", [1])
    assert column.name == "col1"

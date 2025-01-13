from safeds.data.tabular.containers import Column


def test_should_return_the_name() -> None:
    column = Column("a", [])
    assert column.name == "a"

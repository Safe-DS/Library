from safeds.data.tabular.containers import Column


def test_iter() -> None:
    column = Column("testColumn", [0, "1"])
    assert list(column) == [0, "1"]

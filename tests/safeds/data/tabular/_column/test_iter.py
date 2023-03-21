from safeds.data.tabular import Column


def test_iter() -> None:
    column = Column([0, "1"], "testColumn")
    assert list(column) == [0, "1"]

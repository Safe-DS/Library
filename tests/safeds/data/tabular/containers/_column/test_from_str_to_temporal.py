from safeds.data.tabular.containers import Column


def test_temporal_column() -> None:
    col = Column("dates", ["01:01:2021", "01:01:2022", "01:01:2023", "01:01:2024"])
    temp_col = col.from_str_to_temporal("%d:%m:%Y")
    assert temp_col.is_temporal

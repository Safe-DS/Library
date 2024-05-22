from safeds.data.tabular.containers import Column


def test_temporal_column() -> None:
    col = Column("dates", ["01:01:2021", "01:01:2022", "01:01:2023", "01:01:2024"])
    temp_col = col.temporal.from_string("%d:%m:%Y")
    assert temp_col.is_temporal

    column = Column("dates", ["01:01:2021", "01:01:2022", "01:01:2023", "01:01:2024"])
    column = column.temporal.from_string("%d:%m:%Y")
    assert not column.temporal.to_string("%Y/%m:%d").is_temporal

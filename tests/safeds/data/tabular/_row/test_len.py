from safeds.data.tabular import Row
from safeds.data.tabular.typing import StringColumnType, IntColumnType, TableSchema


def test_count() -> None:
    row = Row(
        [0, "1"],
        TableSchema({"testColumn1": IntColumnType(), "testColumn2": StringColumnType()})
    )
    assert len(row) == 2

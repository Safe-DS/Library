from safeds.data.tabular import Row
from safeds.data.tabular.typing import IntColumnType, StringColumnType, TableSchema


def test_iter() -> None:
    row = Row(
        [0, "1"],
        TableSchema(
            {"testColumn1": IntColumnType(), "testColumn2": StringColumnType()}
        ),
    )
    assert list(row) == ["testColumn1", "testColumn2"]

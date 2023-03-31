from safeds.data.tabular.containers import Row
from safeds.data.tabular.typing import Int, String, TableSchema


def test_iter() -> None:
    row = Row(
        [0, "1"],
        TableSchema(
            {"testColumn1": Int(), "testColumn2": String()}
        ),
    )
    assert list(row) == ["testColumn1", "testColumn2"]

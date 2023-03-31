from safeds.data.tabular.containers import Row
from safeds.data.tabular.typing import Integer, String, TableSchema


def test_count() -> None:
    row = Row(
        [0, "1"],
        TableSchema(
            {"testColumn1": Integer(), "testColumn2": String()}
        ),
    )
    assert len(row) == 2

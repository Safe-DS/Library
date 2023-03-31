from safeds.data.tabular.containers import Row
from safeds.data.tabular.typing import Integer, Schema, String


def test_count() -> None:
    row = Row(
        [0, "1"],
        Schema({"testColumn1": Integer(), "testColumn2": String()}),
    )
    assert row.count() == 2

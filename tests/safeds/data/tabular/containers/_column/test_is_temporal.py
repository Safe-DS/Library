from datetime import UTC, datetime

import pytest
from safeds.data.tabular.containers import Column

now = datetime.now(UTC)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (Column("a", []), False),
        (Column("a", [now]), True),
        (Column("a", [now.date()]), True),
        (Column("a", [now.time()]), True),
        (Column("a", [now, None]), True),
        (Column("a", ["a", "b"]), False),
    ],
    ids=[
        "empty",
        "datetime",
        "date",
        "time",
        "operator with missing",
        "non-operator",
    ],
)
def test_should_return_whether_column_is_temporal(column: Column, expected: bool) -> None:
    assert column.is_temporal == expected

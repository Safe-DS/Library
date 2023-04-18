from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import Integer, RealNumber, Schema


def test_table_equals_valid() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    schema_expected = Schema(
        {
            "A": Integer(),
            "B": Integer(),
        },
    )

    assert table.schema == schema_expected


def test_table_equals_invalid() -> None:
    table = Table.from_dict({"A": [1], "B": [2]})
    schema_not_expected = Schema(
        {
            "A": RealNumber(),
            "C": Integer(),
        },
    )

    assert table.schema != schema_not_expected

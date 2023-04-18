from safeds.data.tabular.containers import Table

from tests.helpers import resolve_resource_path


def test_has_column_positive() -> None:
    table = Table.from_dict({
        "A": [1],
        "B": [2]
    })
    assert table.has_column("A")


def test_has_column_negative() -> None:
    table = Table.from_dict(
        {
            "A": [1],
            "B": [2]
        }
    )
    assert not table.has_column("C")

from safeds.data.tabular.containers import Table


def test_has_column_true() -> None:
    table = Table.from_dict(
        {
            "A": [1],
            "B": [2]
        }
    )

    assert table.schema.has_column("A")


def test_has_column_false() -> None:
    table = Table.from_dict(
        {
            "A": [1],
            "B": [2]
        }
    )

    assert not table.schema.has_column("C")

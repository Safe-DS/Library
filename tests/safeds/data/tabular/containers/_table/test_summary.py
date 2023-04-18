from safeds.data.tabular.containers import Table


def test_summary() -> None:
    table = Table.from_dict({"col1": [1, 2, 1], "col2": ["a", "b", "c"]})

    truth = Table.from_dict(
        {
            "metrics": [
                "maximum",
                "minimum",
                "mean",
                "mode",
                "median",
                "sum",
                "variance",
                "standard deviation",
                "idness",
                "stability",
                "row count",
            ],
            "col1": [
                "2",
                "1",
                str(4.0 / 3),
                "[1]",
                "1.0",
                "4",
                str(1.0 / 3),
                str(table._data["col1"].std()),
                str(2.0 / 3),
                str(2.0 / 3),
                "3",
            ],
            "col2": [
                "-",
                "-",
                "-",
                "['a', 'b', 'c']",
                "-",
                "-",
                "-",
                "-",
                "1.0",
                str(1.0 / 3),
                "3",
            ],
        },
    )
    assert truth == table.summary()

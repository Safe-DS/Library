import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table({}),
            "++\n++\n++",
        ),
        (
            Table({"col1": [], "col2": []}),
            "+------+------+\n| col1 | col2 |\n| ---  | ---  |\n| null | null |\n+=============+\n+------+------+",
        ),
        (
            Table({"col1": [1, 2], "col2": [3, 4]}),
            "+------+------+\n"
            "| col1 | col2 |\n"
            "|  --- |  --- |\n"
            "|  i64 |  i64 |\n"
            "+=============+\n"
            "|    1 |    3 |\n"
            "|    2 |    4 |\n"
            "+------+------+",
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
def test_should_return_a_string_representation(table: Table, expected: str) -> None:
    assert str(table) == expected

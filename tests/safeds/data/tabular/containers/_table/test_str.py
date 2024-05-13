import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            "+------+------+\n"
            "| col1 | col2 |\n"
            "|  --- |  --- |\n"
            "|  i64 |  i64 |\n"
            "+=============+\n"
            "|    1 |    1 |\n"
            "|    2 |    2 |\n"
            "|    1 |    4 |\n"
            "+------+------+",
        ),
        (
            Table({"col1": [], "col2": []}),
            "+------+------+\n"
            "| col1 | col2 |\n"
            "| ---  | ---  |\n"
            "| null | null |\n"
            "+=============+\n"
            "+------+------+",
        ),
        (
            Table(),
            "++\n++\n++",
        ),
        (
            Table({"col1": [1], "col2": [1]}),
            "+------+------+\n"
            "| col1 | col2 |\n"
            "|  --- |  --- |\n"
            "|  i64 |  i64 |\n"
            "+=============+\n"
            "|    1 |    1 |\n"
            "+------+------+",
        ),
    ],
    ids=["multiple rows", "rowless table", "empty table", "one row"],
)
def test_should_return_a_string_representation(table: Table, expected: str) -> None:
    assert str(table) == expected

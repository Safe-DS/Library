import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError, DuplicateColumnError


@pytest.mark.parametrize(
    ("table_factory", "old_name", "new_name", "expected"),
    [
        (
            lambda: Table({"A": [1], "B": [2]}),
            "A",
            "C",
            Table({"C": [1], "B": [2]}),
        ),
        (
            lambda: Table({"A": [1], "B": [2]}),
            "A",
            "A",
            Table({"A": [1], "B": [2]}),
        ),
    ],
    ids=["name changed", "name unchanged"],
)
class TestHappyPath:
    def test_should_rename_column(
        self,
        table_factory: callable,
        old_name: str,
        new_name: str,
        expected: Table,
    ) -> None:
        actual = table_factory().rename_column(old_name, new_name)
        assert actual.schema == expected.schema
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: callable,
        old_name: str,
        new_name: str,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original: Table = table_factory()
        original.rename_column(old_name, new_name)
        assert original == table_factory()


def test_should_raise_if_old_column_does_not_exist() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).rename_column("A", "B")


def test_should_raise_if_new_column_exists_already() -> None:
    table: Table = Table({"A": [1], "B": [2]})
    with pytest.raises(DuplicateColumnError):
        table.rename_column("A", "B")

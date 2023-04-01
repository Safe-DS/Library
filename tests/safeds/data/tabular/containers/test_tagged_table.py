import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import UnknownColumnNameError


@pytest.fixture()
def table() -> Table:
    return Table.from_columns(
        [
            Column("A", [1, 4]),
            Column("B", [2, 5]),
            Column("C", [3, 6]),
            Column("T", [0, 1]),
        ],
    )


@pytest.fixture()
def tagged_table(table: Table) -> TaggedTable:
    return table.tag_columns(target_name="T")


class TestInit:
    def test_should_raise_if_a_feature_does_not_exist(self, table: Table) -> None:
        with pytest.raises(UnknownColumnNameError):
            table.tag_columns(target_name="T", feature_names=["A", "B", "C", "D"])

    def test_should_raise_if_target_does_not_exist(self, table: Table) -> None:
        with pytest.raises(UnknownColumnNameError):
            table.tag_columns(target_name="D")

    def test_should_raise_if_features_and_target_overlap(self, table: Table) -> None:
        with pytest.raises(ValueError):
            table.tag_columns(target_name="A", feature_names=["A", "B", "C"])

    def test_should_raise_if_features_are_empty_explicitly(self, table: Table) -> None:
        with pytest.raises(ValueError):
            table.tag_columns(target_name="A", feature_names=[])

    def test_should_raise_if_features_are_empty_implicitly(self, table: Table) -> None:
        table = Table.from_columns([Column("A", [1, 4])])

        with pytest.raises(ValueError):
            table.tag_columns(target_name="A")


class TestFeatures:
    def test_should_return_features(self, tagged_table: TaggedTable) -> None:
        assert tagged_table.features == Table.from_columns(
            [
                Column("A", [1, 4]),
                Column("B", [2, 5]),
                Column("C", [3, 6]),
            ],
        )


class TestTarget:
    def test_should_return_target(self, tagged_table: TaggedTable) -> None:
        assert tagged_table.target == Column("T", [0, 1])

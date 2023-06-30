import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import UnknownColumnNameError


@pytest.fixture()
def data() -> dict[str, list[int]]:
    return {
        "A": [1, 4],
        "B": [2, 5],
        "C": [3, 6],
        "T": [0, 1],
    }


@pytest.fixture()
def table(data: dict[str, list[int]]) -> Table:
    return Table(data)


@pytest.fixture()
def tagged_table(table: Table) -> TaggedTable:
    return table.tag_columns(target_name="T")


class TestFromTable:
    def test_should_raise_if_a_feature_does_not_exist(self, table: Table) -> None:
        with pytest.raises(UnknownColumnNameError):
            TaggedTable._from_table(table, target_name="T", feature_names=["A", "B", "C", "D"])

    def test_should_raise_if_target_does_not_exist(self, table: Table) -> None:
        with pytest.raises(UnknownColumnNameError):
            TaggedTable._from_table(table, target_name="D")

    def test_should_raise_if_features_and_target_overlap(self, table: Table) -> None:
        with pytest.raises(ValueError, match="Column 'A' cannot be both feature and target."):
            TaggedTable._from_table(table, target_name="A", feature_names=["A", "B", "C"])

    def test_should_raise_if_features_are_empty_explicitly(self, table: Table) -> None:
        with pytest.raises(ValueError, match="At least one feature column must be specified."):
            TaggedTable._from_table(table, target_name="A", feature_names=[])

    def test_should_raise_if_features_are_empty_implicitly(self) -> None:
        table = Table({"A": [1, 4]})

        with pytest.raises(ValueError, match="At least one feature column must be specified."):
            TaggedTable._from_table(table, target_name="A")


class TestInit:
    def test_should_raise_if_a_feature_does_not_exist(self, data: dict[str, list[int]]) -> None:
        with pytest.raises(UnknownColumnNameError):
            TaggedTable(data, target_name="T", feature_names=["A", "B", "C", "D"])

    def test_should_raise_if_target_does_not_exist(self, data: dict[str, list[int]]) -> None:
        with pytest.raises(UnknownColumnNameError):
            TaggedTable(data, target_name="D")

    def test_should_raise_if_features_and_target_overlap(self, data: dict[str, list[int]]) -> None:
        with pytest.raises(ValueError, match="Column 'A' cannot be both feature and target."):
            TaggedTable(data, target_name="A", feature_names=["A", "B", "C"])

    def test_should_raise_if_features_are_empty_explicitly(self, data: dict[str, list[int]]) -> None:
        with pytest.raises(ValueError, match="At least one feature column must be specified."):
            TaggedTable(data, target_name="A", feature_names=[])

    def test_should_raise_if_features_are_empty_implicitly(self) -> None:
        data = {"A": [1, 4]}

        with pytest.raises(ValueError, match="At least one feature column must be specified."):
            TaggedTable(data, target_name="A")


class TestFeatures:
    def test_should_return_features(self, tagged_table: TaggedTable) -> None:
        assert tagged_table.features == Table(
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
            },
        )


class TestTarget:
    def test_should_return_target(self, tagged_table: TaggedTable) -> None:
        assert tagged_table.target == Column("T", [0, 1])


class TestCopy:
    @pytest.mark.parametrize(
        "tagged_table",
        [
            TaggedTable({"a": [], "b": []}, target_name="b", feature_names=["a"]),
            TaggedTable({"a": ["a", 3, 0.1], "b": [True, False, None]}, target_name="b", feature_names=["a"]),
        ],
        ids=["empty-rows", "normal"],
    )
    def test_should_copy_tagged_table(self, tagged_table: TaggedTable) -> None:
        copied = tagged_table._copy()
        assert copied == tagged_table
        assert copied is not tagged_table

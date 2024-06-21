import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import KNearestNeighborsImputer
from safeds.exceptions import ColumnNotFoundError, TransformerNotFittedError


class TestInit:
    def test_should_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match='Parameter "neighbor_count" must be greater than 0.'):
            _ = KNearestNeighborsImputer(neighbor_count=0)

    def test_should_raise_type_error(self) -> None:
        with pytest.raises(TypeError, match='Parameter "neighbor_count" must be a integer.'):
            _ = KNearestNeighborsImputer(neighbor_count=1.5)


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            KNearestNeighborsImputer(column_names=["col2", "col3"]).fit(table) 
    
    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The KNearestNeighborsImputer cannot be fitted because the table contains 0 rows"):
            KNearestNeighborsImputer().fit(Table({"col1": []}))

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = KNearestNeighborsImputer()
        transformer.fit(table)

        assert transformer._column_names is None
        assert transformer._wrapped_transformer is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
                "col2": [5.0, 50.0, 100.0],
            },
        )

        transformer = KNearestNeighborsImputer()
        transformer.fit(table_to_fit)

        table_to_transform = Table(
            {
                "col3": ["a", "b", "c"],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            transformer.transform(table_to_transform)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = KNearestNeighborsImputer()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)

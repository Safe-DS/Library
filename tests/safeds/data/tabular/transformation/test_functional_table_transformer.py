import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import FunctionalTableTransformer
from safeds.exceptions import ColumnNotFoundError


def valid_callable(table: Table) -> Table:
    return table.remove_columns(["col1"])


class TestInit:
    def test_should_not_raise_type_error(self) -> None:
        FunctionalTableTransformer(valid_callable)


class TestFit:
    def test_should_return_self(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(valid_callable)
        assert transformer.fit(table) is transformer


class TestIsFitted:
    def test_should_always_be_fitted(self) -> None:
        transformer = FunctionalTableTransformer(valid_callable)
        assert transformer.is_fitted


class TestTransform:
    def test_should_raise_specific_error_when_error_in_method(self) -> None:
        table = Table(
            {
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(valid_callable)
        with pytest.raises(ColumnNotFoundError):
            transformer.transform(table)

    def test_should_not_modify_original_table(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(valid_callable)
        transformer.transform(table)
        assert table == Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )

    def test_should_return_modified_table(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(valid_callable)
        transformed_table = transformer.transform(table)
        assert transformed_table == Table(
            {
                "col2": [1, 2, 3],
            },
        )


class TestFitAndTransform:
    def test_should_return_self(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(valid_callable)
        assert transformer.fit_and_transform(table)[0] is transformer

    def test_should_not_modify_original_table(self) -> None:
        table = Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )
        transformer = FunctionalTableTransformer(valid_callable)
        transformer.fit_and_transform(table)
        assert table == Table(
            {
                "col1": [1, 2, 3],
                "col2": [1, 2, 3],
            },
        )

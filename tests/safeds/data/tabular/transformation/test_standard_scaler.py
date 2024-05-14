import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import StandardScaler
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError, TransformerNotFittedError

from tests.helpers import assert_tables_equal


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            StandardScaler().fit(table, ["col2", "col3"])

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            StandardScaler().fit(
                Table({"col1": ["one", "two", "apple"], "col2": ["three", "four", "banana"]}),
                ["col1", "col2"],
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The StandardScaler cannot be fitted because the table contains 0 rows"):
            StandardScaler().fit(Table({"col1": []}), None)

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()
        transformer.fit(table, None)

        assert transformer._column_names is None
        assert transformer._data_mean is None
        assert transformer._data_standard_deviation is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
                "col2": [5.0, 50.0, 100.0],
            },
        )

        transformer = StandardScaler().fit(table_to_fit, None)

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

        transformer = StandardScaler()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            StandardScaler().fit(Table({"col1": [1, 2, 3], "col2": [2, 3, 4]}), ["col1", "col2"]).transform(
                Table({"col1": ["a", "b", "c"], "col2": ["b", "c", "e"]}),
            )


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = StandardScaler()
        assert not transformer.is_fitted

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted


class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 0.0, 1.0, 1.0],
                        "col2": [0.0, 0.0, 1.0, 1.0],
                    },
                ),
                None,
                Table(
                    {
                        "col1": [-1.0, -1.0, 1.0, 1.0],
                        "col2": [-1.0, -1.0, 1.0, 1.0],
                    },
                ),
            ),
        ],
        ids=["two_columns"],
    )
    def test_should_return_fitted_transformer_and_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = StandardScaler().fit_and_transform(table, column_names)
        assert fitted_transformer.is_fitted
        assert_tables_equal(transformed_table, expected)

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        StandardScaler().fit_and_transform(table)

        expected = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        assert table == expected


class TestInverseTransform:
    @pytest.mark.parametrize(
        "table",
        [
            Table(
                {
                    "col1": [0.0, 5.0, 5.0, 10.0],
                },
            ),
        ],
        ids=["one_column"],
    )
    def test_should_return_original_table(self, table: Table) -> None:
        transformer = StandardScaler().fit(table, None)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        transformer = StandardScaler().fit(table, None)
        transformed_table = transformer.transform(table)
        transformed_table = transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        assert_tables_equal(transformed_table, expected)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.inverse_transform(table)

    def test_should_raise_if_column_not_found(self) -> None:
        with pytest.raises(ColumnNotFoundError):
            StandardScaler().fit(Table({"col1": [1, 2, 4], "col2": [2, 3, 4]}), ["col1", "col2"]).inverse_transform(
                Table({"col3": [0, 1, 2]}),
            )

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            StandardScaler().fit(Table({"col1": [1, 2, 4], "col2": [2, 3, 4]}), ["col1", "col2"]).inverse_transform(
                Table({"col1": ["one", "two", "apple"], "col2": ["three", "four", "banana"]}),
            )

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RobustScaler
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
            RobustScaler(column_names=["col2", "col3"]).fit(table)

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            RobustScaler(column_names=["col1", "col2"]).fit(
                Table({"col1": ["one", "two", "apple"], "col2": ["three", "four", "banana"]}),
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The RobustScaler cannot be fitted because the table contains 0 rows"):
            RobustScaler().fit(Table({"col1": []}))

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = RobustScaler()
        transformer.fit(table)

        assert transformer._column_names is None
        assert transformer._data_median is None
        assert transformer._data_scale is None

    # TODO: Tests for None and NaN values should be moved to their own function
    def test_should_not_divide_by_zero(self) -> None:
        table = Table(
            {
                "col1": [1.0, 1.0, 2.0, 1.0],
                "col2": [3.0, 3.0, 3.0, 3.0],
                #"col3": [1.0, float("nan"), float("nan"), float("nan")],
                "col4": [1.0, None, None, None],
            },
        )
        target = Table(
            {
                "col1": [0.0, 0.0, 1.0, 0.0],
                "col2": [0.0, 0.0, 0.0, 0.0],
                #"col3": [0.0, float("nan"), float("nan"), float("nan")],
                "col4": [0.0, None, None, None],
            },
        )
        transformer = RobustScaler()
        f_transformer = transformer.fit(table)
        table = f_transformer.transform(table)
        assert(table == target)



class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
                "col2": [5.0, 50.0, 100.0],
            },
        )

        transformer = RobustScaler().fit(table_to_fit)

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

        transformer = RobustScaler()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            RobustScaler(column_names=["col1", "col2"]).fit(Table({"col1": [1, 2, 3], "col2": [2, 3, 4]})).transform(
                Table({"col1": ["a", "b", "c"], "col2": ["b", "c", "e"]}),
            )


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = RobustScaler()
        assert not transformer.is_fitted

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = RobustScaler()
        fitted_transformer = transformer.fit(table)
        assert fitted_transformer.is_fitted

class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": [1.0, 2.0, 3.0, 4.0],
                        "col2": [1.0, 2.0, 3.0, 4.0],
                    },
                ),
                None,
                Table(
                    {
                        "col1": [-1.5, -0.5, 0.5, 1.5],
                        "col2": [-1.5, -0.5, 0.5, 1.5],
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
        fitted_transformer, transformed_table = RobustScaler(column_names=column_names).fit_and_transform(table)
        assert fitted_transformer.is_fitted
        assert_tables_equal(transformed_table, expected)

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        RobustScaler().fit_and_transform(table)

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
                    "col1": [1.0, 2.0, 3.0, 4.0],
                },
            ),
        ],
        ids=["one_column"],
    )
    def test_should_return_original_table(self, table: Table) -> None:
        transformer = RobustScaler().fit(table)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 0.5, 1.0, 1.5, 2.0],
            },
        )

        transformer = RobustScaler().fit(table)
        transformed_table = transformer.transform(table)
        transformed_table = transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1": [0.0, 0.5, 1.0, 1.5, 2.0],
            },
        )

        assert_tables_equal(transformed_table, expected)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
            },
        )

        transformer = RobustScaler()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.inverse_transform(table)

    def test_should_raise_if_column_not_found(self) -> None:
        with pytest.raises(ColumnNotFoundError):
            RobustScaler(column_names=["col1", "col2"]).fit(
                Table({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]}),
            ).inverse_transform(
                Table({"col3": [0, 1, 2, 3]}),
            )

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            RobustScaler(column_names=["col1", "col2"]).fit(
                Table({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]}),
            ).inverse_transform(
                Table({"col1": ["one", "two", "apple"], "col2": ["three", "four", "banana"]}),
            )

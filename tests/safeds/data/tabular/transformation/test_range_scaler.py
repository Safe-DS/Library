import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError, TransformerNotFittedError


class TestInit:
    def test_should_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match='Parameter "max_" must be greater than parameter "min_".'):
            _ = RangeScaler(min_=10, max_=0)


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            RangeScaler(column_names=["col2", "col3"]).fit(table)

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            RangeScaler(column_names=["col1", "col2"]).fit(Table({"col1": ["a", "b"], "col2": [1, "c"]}))

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The RangeScaler cannot be fitted because the table contains 0 rows"):
            RangeScaler().fit(Table({"col1": []}))

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = RangeScaler()
        transformer.fit(table)

        assert transformer._column_names is None
        assert transformer._data_min is None
        assert transformer._data_max is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
                "col2": [5.0, 50.0, 100.0],
            },
        )

        transformer = RangeScaler().fit(table_to_fit)

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

        transformer = RangeScaler()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            RangeScaler(column_names=["col1", "col2"]).fit(Table({"col1": [1, 2, 3], "col2": [2, 3, 4]})).transform(
                Table({"col1": ["a", "b", "c"], "col2": ["c", "d", "e"]}),
            )


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = RangeScaler()
        assert not transformer.is_fitted

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = RangeScaler()
        fitted_transformer = transformer.fit(table)
        assert fitted_transformer.is_fitted


class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                None,
                Table(
                    {
                        "col1": [0.0, 0.5, 0.5, 1.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                ["col1"],
                Table(
                    {
                        "col1": [0.0, 0.5, 0.5, 1.0],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
            ),
        ],
        ids=["one_column", "two_columns"],
    )
    def test_should_return_fitted_transformer_and_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = RangeScaler(column_names=column_names).fit_and_transform(table)
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                None,
                Table(
                    {
                        "col1": [-10.0, 0.0, 0.0, 10.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                ["col1"],
                Table(
                    {
                        "col1": [-10.0, 0.0, 0.0, 10.0],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
            ),
        ],
        ids=["one_column", "two_columns"],
    )
    def test_should_return_fitted_transformer_and_transformed_table_with_correct_range(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = RangeScaler(
            min_=-10.0,
            max_=10.0,
            column_names=column_names,
        ).fit_and_transform(
            table,
        )
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        RangeScaler().fit_and_transform(table)

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
    )
    def test_should_return_original_table(self, table: Table) -> None:
        transformer = RangeScaler().fit(table)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        transformer = RangeScaler().fit(table)
        transformed_table = transformer.transform(table)
        transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        assert transformed_table == expected

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 5.0, 10.0],
            },
        )

        transformer = RangeScaler()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.inverse_transform(table)

    def test_should_raise_if_column_not_found(self) -> None:
        with pytest.raises(ColumnNotFoundError):
            RangeScaler(column_names=["col1", "col2"]).fit(
                Table({"col1": [1, 2, 3], "col2": [2, 3, 4]}),
            ).inverse_transform(
                Table({"col3": [1, 2, 3]}),
            )

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            RangeScaler(column_names=["col1", "col2"]).fit(
                Table({"col1": [1, 2, 3], "col2": [2, 3, 4]}),
            ).inverse_transform(
                Table({"col1": ["1", "2", "three"], "col2": [1, 2, "four"]}),
            )

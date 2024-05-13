import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, ColumnNotFoundError


class TestInit:
    def test_should_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match='Parameter "maximum" must be higher than parameter "minimum".'):
            _ = RangeScaler(min_=10, max_=0)


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(ColumnNotFoundError, match=r"Could not find column\(s\) 'col2, col3'"):
            RangeScaler().fit(table, ["col2", "col3"])

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(
            NonNumericColumnError,
            match=r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1', 'col2'\]",
        ):
            RangeScaler().fit(Table({"col1": ["a", "b"], "col2": [1, "c"]}), ["col1", "col2"])

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The RangeScaler cannot be fitted because the table contains 0 rows"):
            RangeScaler().fit(Table({"col1": []}), ["col1"])

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = RangeScaler()
        transformer.fit(table, None)

        assert transformer._wrapped_transformer is None
        assert transformer._column_names is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
                "col2": [5.0, 50.0, 100.0],
            },
        )

        transformer = RangeScaler().fit(table_to_fit, None)

        table_to_transform = Table(
            {
                "col3": ["a", "b", "c"],
            },
        )

        with pytest.raises(ColumnNotFoundError, match=r"Could not find column\(s\) 'col1, col2'"):
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
        with pytest.raises(
            NonNumericColumnError,
            match=r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1', 'col2'\]",
        ):
            RangeScaler().fit(Table({"col1": [1, 2, 3], "col2": [2, 3, 4]}), ["col1", "col2"]).transform(
                Table({"col1": ["a", "b", "c"], "col2": ["c", "d", "e"]}),
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The RangeScaler cannot transform the table because it contains 0 rows"):
            RangeScaler().fit(Table({"col1": [1, 2, 3]}), ["col1"]).transform(Table({"col1": []}))


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
        fitted_transformer = transformer.fit(table, None)
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
        fitted_transformer, transformed_table = RangeScaler().fit_and_transform(table, column_names)
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
        fitted_transformer, transformed_table = RangeScaler(min_=-10.0, max_=10.0).fit_and_transform(
            table,
            column_names,
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

    def test_get_names_of_added_columns(self) -> None:
        transformer = RangeScaler()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_added_columns()

        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_added_columns() == []

    def test_get_names_of_changed_columns(self) -> None:
        transformer = RangeScaler()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_changed_columns()
        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_changed_columns() == ["a"]

    def test_get_names_of_removed_columns(self) -> None:
        transformer = RangeScaler()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_removed_columns()

        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_removed_columns() == []


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
        transformer = RangeScaler().fit(table, None)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        transformer = RangeScaler().fit(table, None)
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
        with pytest.raises(ColumnNotFoundError, match=r"Could not find column\(s\) 'col1, col2'"):
            RangeScaler().fit(Table({"col1": [1, 2, 3], "col2": [2, 3, 4]}), ["col1", "col2"]).inverse_transform(
                Table({"col3": [1, 2, 3]}),
            )

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(
            NonNumericColumnError,
            match=r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1', 'col2'\]",
        ):
            RangeScaler().fit(Table({"col1": [1, 2, 3], "col2": [2, 3, 4]}), ["col1", "col2"]).inverse_transform(
                Table({"col1": ["1", "2", "three"], "col2": [1, 2, "four"]}),
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The RangeScaler cannot transform the table because it contains 0 rows"):
            RangeScaler().fit(Table({"col1": [1, 2, 3]}), ["col1"]).inverse_transform(Table({"col1": []}))

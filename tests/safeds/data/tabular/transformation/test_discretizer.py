import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import Discretizer
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, UnknownColumnNameError


class TestInit:
    def test_should_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Parameter 'number_of_bins' must be >= 2."):
            _ = Discretizer(1)


class TestFit:
    @pytest.mark.parametrize(
        ("table", "columns", "error", "error_message"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                ["col2"],
                UnknownColumnNameError,
                r"Could not find column\(s\) 'col2'",
            ),
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                        "col3": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                ["col4", "col5"],
                UnknownColumnNameError,
                r"Could not find column\(s\) 'col4, col5'",
            ),
            (Table(), ["col2"], ValueError, "The Discretizer cannot be fitted because the table contains 0 rows"),
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                        "col2": ["a", "b", "c", "d"],
                    },
                ),
                ["col2"],
                NonNumericColumnError,
                "Tried to do a numerical operation on one or multiple non-numerical columns: \ncol2 is of type String.",
            ),
        ],
        ids=["UnknownColumnNameError", "multiple missing columns", "ValueError", "NonNumericColumnError"],
    )
    def test_should_raise_errors(self, table: Table, columns: list[str], error: type[Exception], error_message: str) -> None:
        with pytest.raises(error, match=error_message):
            Discretizer().fit(table, columns)

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()
        transformer.fit(table, None)

        assert transformer._wrapped_transformer is None
        assert transformer._column_names is None


class TestTransform:
    @pytest.mark.parametrize(
        ("table_to_transform", "columns", "error", "error_message"),
        [
            (
                Table(
                    {
                        "col2": ["a", "b", "c"],
                    },
                ),
                ["col1"],
                UnknownColumnNameError,
                r"Could not find column\(s\) 'col1'",
            ),
            (
                Table(
                    {
                        "col2": ["a", "b", "c"],
                    },
                ),
                ["col3", "col1"],
                UnknownColumnNameError,
                r"Could not find column\(s\) 'col3, col1'",
            ),
            (Table(), ["col1", "col3"], ValueError, "The table cannot be transformed because it contains 0 rows"),
            (
                Table(
                    {
                        "col1": ["a", "b", "c", "d"],
                    },
                ),
                ["col1"],
                NonNumericColumnError,
                "Tried to do a numerical operation on one or multiple non-numerical columns: \ncol1 is of type String.",
            ),
        ],
        ids=["UnknownColumnNameError", "multiple missing columns", "ValueError", "NonNumericColumnError"],
    )
    def test_should_raise_errors(self, table_to_transform: Table, columns: list[str], error: type[Exception], error_message: str) -> None:
        table_to_fit = Table(
            {
                "col1": [0.0, 5.0, 10.0],
                "col3": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer().fit(table_to_fit, columns)

        with pytest.raises(error, match=error_message):
            transformer.transform(table_to_transform)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = Discretizer()
        assert not transformer.is_fitted()

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted()


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
                        "col1": [0.0, 2.0, 2.0, 3.0],
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
                        "col1": [0.0, 2.0, 2.0, 3.0],
                        "col2": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
            ),
        ],
        ids=["None", "col1"],
    )
    def test_should_return_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        assert Discretizer().fit_and_transform(table, column_names) == expected

    @pytest.mark.parametrize(
        ("table", "number_of_bins", "expected"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                2,
                Table(
                    {
                        "col1": [0, 1.0, 1.0, 1.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "col1": [0.0, 5.0, 5.0, 10.0],
                    },
                ),
                10,
                Table(
                    {
                        "col1": [0.0, 4.0, 4.0, 7.0],
                    },
                ),
            ),
        ],
        ids=["2", "10"],
    )
    def test_should_return_transformed_table_with_correct_number_of_bins(
        self,
        table: Table,
        number_of_bins: int,
        expected: Table,
    ) -> None:
        assert Discretizer(number_of_bins).fit_and_transform(table, ["col1"]) == expected

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        Discretizer().fit_and_transform(table)

        expected = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        assert table == expected

    def test_get_names_of_added_columns(self) -> None:
        transformer = Discretizer()
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
        transformer = Discretizer()
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
        transformer = Discretizer()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_removed_columns()

        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_removed_columns() == []

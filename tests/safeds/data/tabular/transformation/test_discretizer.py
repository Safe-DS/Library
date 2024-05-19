import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import Discretizer
from safeds.exceptions import ColumnNotFoundError, NonNumericColumnError, OutOfBoundsError, TransformerNotFittedError


class TestInit:
    def test_should_raise_value_error(self) -> None:
        with pytest.raises(OutOfBoundsError):
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
                ColumnNotFoundError,
                None,
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
                ColumnNotFoundError,
                None,
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
        ids=["ColumnNotFoundError", "multiple missing columns", "ValueError", "NonNumericColumnError"],
    )
    def test_should_raise_errors(
        self,
        table: Table,
        columns: list[str],
        error: type[Exception],
        error_message: str | None,
    ) -> None:
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
                ColumnNotFoundError,
                None,
            ),
            (
                Table(
                    {
                        "col2": ["a", "b", "c"],
                    },
                ),
                ["col3", "col1"],
                ColumnNotFoundError,
                None,
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
        ids=["ColumnNotFoundError", "multiple missing columns", "ValueError", "NonNumericColumnError"],
    )
    def test_should_raise_errors(
        self,
        table_to_transform: Table,
        columns: list[str],
        error: type[Exception],
        error_message: str | None,
    ) -> None:
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

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = Discretizer()
        assert not transformer.is_fitted

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = Discretizer()
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
    def test_should_return_fitted_transformer_and_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = Discretizer().fit_and_transform(table, column_names)
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

    @pytest.mark.parametrize(
        ("table", "bin_count", "expected"),
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
        bin_count: int,
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = Discretizer(bin_count).fit_and_transform(table, ["col1"])
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

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

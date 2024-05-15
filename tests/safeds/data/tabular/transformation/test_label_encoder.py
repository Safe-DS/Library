import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import LabelEncoder
from safeds.exceptions import ColumnNotFoundError, ColumnTypeError, TransformerNotFittedError


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            LabelEncoder().fit(table, ["col2", "col3"])

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The LabelEncoder cannot be fitted because the table contains 0 rows"):
            LabelEncoder().fit(Table({"col1": []}), ["col1"])

    def test_should_warn_if_table_contains_numerical_data(self) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                r"The columns \['col1'\] contain numerical data. The LabelEncoder is designed to encode non-numerical"
                r" values into numerical values"
            ),
        ):
            LabelEncoder().fit(Table({"col1": [1, 2]}), ["col1"])

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = LabelEncoder()
        transformer.fit(table, None)

        assert transformer._column_names is None
        assert transformer._mapping is None
        assert transformer._inverse_mapping is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": ["a", "b", "c"],
                "col2": ["d", "e", "f"],
            },
        )

        transformer = LabelEncoder().fit(table_to_fit, None)

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
                "col1": ["a", "b", "c"],
            },
        )

        transformer = LabelEncoder()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = LabelEncoder()
        assert not transformer.is_fitted

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = LabelEncoder()
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted


class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table(
                    {
                        "col1": ["a", "b", "b", "c"],
                    },
                ),
                None,
                Table(
                    {
                        "col1": [0.0, 1.0, 1.0, 2.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "col1": ["a", "b", "b", "c"],
                        "col2": ["a", "b", "b", "c"],
                    },
                ),
                ["col1"],
                Table(
                    {
                        "col1": [0.0, 1.0, 1.0, 2.0],
                        "col2": ["a", "b", "b", "c"],
                    },
                ),
            ),
        ],
        ids=["no_column_names", "with_column_names"],
    )
    def test_should_return_fitted_transformer_and_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = LabelEncoder().fit_and_transform(table, column_names)
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        LabelEncoder().fit_and_transform(table)

        expected = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        assert table == expected


class TestInverseTransform:
    @pytest.mark.parametrize(
        "table",
        [
            Table(
                {
                    "col1": ["a", "b", "b", "c"],
                },
            ),
        ],
        ids=["no_column_names"],
    )
    def test_should_return_original_table(self, table: Table) -> None:
        transformer = LabelEncoder().fit(table, None)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = LabelEncoder().fit(table, None)
        transformed_table = transformer.transform(table)
        transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1": [0.0, 1.0, 2.0],
            },
        )

        assert transformed_table == expected

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 1.0, 1.0, 2.0],
            },
        )

        transformer = LabelEncoder()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.inverse_transform(table)

    def test_should_raise_if_column_not_found(self) -> None:
        with pytest.raises(ColumnNotFoundError):
            LabelEncoder().fit(
                Table({"col1": ["one", "two"], "col2": ["three", "four"]}),
                ["col1", "col2"],
            ).inverse_transform(Table({"col3": [1.0, 0.0]}))

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            LabelEncoder().fit(
                Table({"col1": ["one", "two"], "col2": ["three", "four"]}),
                ["col1", "col2"],
            ).inverse_transform(Table({"col1": ["1", "null"], "col2": ["2", "apple"]}))

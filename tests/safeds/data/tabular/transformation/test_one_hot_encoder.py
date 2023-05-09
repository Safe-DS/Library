import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        with pytest.raises(UnknownColumnNameError):
            OneHotEncoder().fit(table, ["col2"])

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = OneHotEncoder()
        transformer.fit(table, None)

        assert transformer._column_names is None
        assert transformer._value_to_column is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = OneHotEncoder().fit(table_to_fit, None)

        table_to_transform = Table(
            {
                "col2": ["a", "b", "c"],
            },
        )

        with pytest.raises(UnknownColumnNameError):
            transformer.transform(table_to_transform)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = OneHotEncoder()
        assert not transformer.is_fitted()

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = OneHotEncoder()
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted()


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
                        "col1_a": [1.0, 0.0, 0.0, 0.0],
                        "col1_b": [0.0, 1.0, 1.0, 0.0],
                        "col1_c": [0.0, 0.0, 0.0, 1.0],
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
                        "col1_a": [1.0, 0.0, 0.0, 0.0],
                        "col1_b": [0.0, 1.0, 1.0, 0.0],
                        "col1_c": [0.0, 0.0, 0.0, 1.0],
                        "col2": ["a", "b", "b", "c"],
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
                ["col1", "col2"],
                Table(
                    {
                        "col1_a": [1.0, 0.0, 0.0, 0.0],
                        "col1_b": [0.0, 1.0, 1.0, 0.0],
                        "col1_c": [0.0, 0.0, 0.0, 1.0],
                        "col2_a": [1.0, 0.0, 0.0, 0.0],
                        "col2_b": [0.0, 1.0, 1.0, 0.0],
                        "col2_c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
            ),
        ],
        ids=["all columns", "one column", "multiple columns"],
    )
    def test_should_return_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        assert OneHotEncoder().fit_and_transform(table, column_names) == expected

    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        OneHotEncoder().fit_and_transform(table)

        expected = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        assert table == expected


class TestInverseTransform:
    @pytest.mark.parametrize(
        ("table_to_fit", "column_names", "table_to_transform"),
        [
            (
                Table(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
                ["b"],
                Table(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
                ["b"],
                Table(
                    {
                        "c": [0.0, 0.0, 0.0, 1.0],
                        "b": ["a", "b", "b", "c"],
                        "a": [1.0, 0.0, 0.0, 0.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "bb": ["a", "b", "b", "c"],
                    },
                ),
                ["b", "bb"],
                Table(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "bb": ["a", "b", "b", "c"],
                    },
                ),
            ),
        ],
        ids=[
            "same table to fit and transform",
            "different tables to fit and transform",
            "one column name is a prefix of another column name",
        ],
    )
    def test_should_return_original_table(
        self,
        table_to_fit: Table,
        column_names: list[str],
        table_to_transform: Table,
    ) -> None:
        transformer = OneHotEncoder().fit(table_to_fit, column_names)

        result = transformer.inverse_transform(transformer.transform(table_to_transform))

        # This checks whether the columns are in the same order
        assert result.column_names == table_to_transform.column_names
        # This is subsumed by the next assertion, but we get a better error message
        assert result.schema == table_to_transform.schema
        assert result == table_to_transform

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "b", "c"],
            },
        )

        transformer = OneHotEncoder().fit(table, None)
        transformed_table = transformer.transform(table)
        transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1_a": [1.0, 0.0, 0.0, 0.0],
                "col1_b": [0.0, 1.0, 1.0, 0.0],
                "col1_c": [0.0, 0.0, 0.0, 1.0],
            },
        )

        assert transformed_table == expected

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.0, 1.0, 1.0, 0.0],
                "c": [0.0, 0.0, 0.0, 1.0],
            },
        )

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError):
            transformer.inverse_transform(table)

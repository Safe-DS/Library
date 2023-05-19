import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError, ValueNotPresentWhenFittedError


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

    def test_should_raise_value_not_present_when_fitted(self) -> None:
        fit_table = Table(
            {
                "col1": ["a"],
            },
        )
        transform_table = Table(
            {
                "col1": ["b"],
            },
        )

        transformer = OneHotEncoder().fit(fit_table, None)

        with pytest.raises(ValueNotPresentWhenFittedError):
            transformer.transform(transform_table)


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
                        "col1__a": [1.0, 0.0, 0.0, 0.0],
                        "col1__b": [0.0, 1.0, 1.0, 0.0],
                        "col1__c": [0.0, 0.0, 0.0, 1.0],
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
                        "col1__a": [1.0, 0.0, 0.0, 0.0],
                        "col1__b": [0.0, 1.0, 1.0, 0.0],
                        "col1__c": [0.0, 0.0, 0.0, 1.0],
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
                        "col1__a": [1.0, 0.0, 0.0, 0.0],
                        "col1__b": [0.0, 1.0, 1.0, 0.0],
                        "col1__c": [0.0, 0.0, 0.0, 1.0],
                        "col2__a": [1.0, 0.0, 0.0, 0.0],
                        "col2__b": [0.0, 1.0, 1.0, 0.0],
                        "col2__c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a_b": ["c"],
                        "a": ["b_c"],
                    },
                ),
                None,
                Table(
                    {
                        "a_b__c": [1.0],
                        "a__b_c": [1.0],
                    },
                ),
            ),
            (
                Table(
                    {
                        "a__b": ["c", "d"],
                        "a": ["b__c", "d"],
                    },
                ),
                None,
                Table(
                    {
                        "a__b__c": [1.0, 0.0],
                        "a__b__d": [0.0, 1.0],
                        "a__b__c#2": [1.0, 0.0],
                        "a__d": [0.0, 1.0],
                    },
                ),
            ),
        ],
        ids=["all columns", "one column", "multiple columns", "single underscore counterexample", "name conflict"],
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

    def test_get_names_of_added_columns(self) -> None:
        transformer = OneHotEncoder()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_added_columns()

        table = Table(
            {
                "a__b": ["c", "d"],
                "a": ["b__c", "d"],
            },
        )
        added_columns = ["a__b__c", "a__b__d", "a__b__c#2", "a__d"]

        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_added_columns() == added_columns

    def test_get_names_of_changed_columns(self) -> None:
        transformer = OneHotEncoder()
        with pytest.warns(
            UserWarning,
            match="OneHotEncoder only removes and adds, but does not change any columns.",
        ), pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_changed_columns()

        with pytest.warns(UserWarning, match="OneHotEncoder only removes and adds, but does not change any columns."):
            table = Table(
                {
                    "a": ["b"],
                },
            )
            transformer = transformer.fit(table, None)
            assert transformer.get_names_of_changed_columns() == []

    def test_get_names_of_removed_columns(self) -> None:
        transformer = OneHotEncoder()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_removed_columns()

        table = Table(
            {
                "a": ["b"],
            },
        )
        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_removed_columns() == ["a"]


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
                "col1__a": [1.0, 0.0, 0.0, 0.0],
                "col1__b": [0.0, 1.0, 1.0, 0.0],
                "col1__c": [0.0, 0.0, 0.0, 1.0],
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

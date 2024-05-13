import warnings

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import (
    NonNumericColumnError,
    TransformerNotFittedError,
    ColumnNotFoundError,
    ValueNotPresentWhenFittedError,
)


class TestEq:

    def test_should_be_not_implemented(self) -> None:
        assert OneHotEncoder().__eq__(Table()) is NotImplemented

    def test_should_be_equal(self) -> None:
        table1 = Table({"a": ["a", "b", "c"], "b": ["a", "b", "c"]})
        table2 = Table({"b": ["a", "b", "c"], "a": ["a", "b", "c"]})
        assert OneHotEncoder().fit(table1, None) == OneHotEncoder().fit(table2, None)

    @pytest.mark.parametrize(
        ("table1", "table2"),
        [
            (Table({"a": ["a", "b", "c"], "b": ["a", "b", "c"]}), Table({"a": ["a", "b", "c"], "aa": ["a", "b", "c"]})),
            (Table({"a": ["a", "b", "c"], "b": ["a", "b", "c"]}), Table({"a": ["a", "b", "c"], "b": ["a", "b", "d"]})),
        ],
    )
    def test_should_be_not_equal(self, table1: Table, table2: Table) -> None:
        assert OneHotEncoder().fit(table1, None) != OneHotEncoder().fit(table2, None)


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        with pytest.raises(ColumnNotFoundError, match=r"Could not find column\(s\) 'col2, col3'"):
            OneHotEncoder().fit(table, ["col2", "col3"])

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The OneHotEncoder cannot be fitted because the table contains 0 rows"):
            OneHotEncoder().fit(Table({"col1": []}), ["col1"])

    def test_should_warn_if_table_contains_numerical_data(self) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                r"The columns \['col1'\] contain numerical data. The OneHotEncoder is designed to encode non-numerical"
                r" values into numerical values"
            ),
        ):
            OneHotEncoder().fit(Table({"col1": [1, 2, 3]}), ["col1"])

    @pytest.mark.parametrize(
        "table",
        [
            Table(
                {
                    "col1": ["a", "b", "c"],
                },
            ),
            Table(
                {
                    "col1": ["a", "b", float("nan")],
                },
            ),
        ],
        ids=["string table", "table with nan"],
    )
    def test_should_not_change_original_transformer(self, table: Table) -> None:
        transformer = OneHotEncoder()
        transformer.fit(table, None)

        assert transformer._column_names is None
        assert transformer._value_to_column is None
        assert transformer._value_to_column_nans is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": ["a", "b", "c"],
                "col2": ["a2", "b2", "c2"],
            },
        )

        transformer = OneHotEncoder().fit(table_to_fit, None)

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
                "col1": ["a", "b", "c"],
            },
        )

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The LabelEncoder cannot transform the table because it contains 0 rows"):
            OneHotEncoder().fit(Table({"col1": ["one", "two", "three"]}), ["col1"]).transform(Table({"col1": []}))

    def test_should_raise_value_not_present_when_fitted(self) -> None:
        fit_table = Table(
            {"col1": ["a"], "col2": ["b"]},
        )
        transform_table = Table(
            {"col1": ["b", "c"], "col2": ["a", "b"]},
        )

        transformer = OneHotEncoder().fit(fit_table, None)

        with pytest.raises(
            ValueNotPresentWhenFittedError,
            match=(
                r"Value\(s\) not present in the table the transformer was fitted on: \nb in column col1\nc in column"
                r" col1\na in column col2"
            ),
        ):
            transformer.transform(transform_table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = OneHotEncoder()
        assert not transformer.is_fitted

    @pytest.mark.parametrize(
        "table",
        [
            Table(
                {
                    "col1": ["a", "b", "c"],
                },
            ),
            Table(
                {
                    "col1": [float("nan")],
                },
            ),
        ],
        ids=["table with strings", "table with nans"],
    )
    def test_should_return_true_after_fitting(self, table: Table) -> None:
        transformer = OneHotEncoder()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=(
                    r"The columns .+ contain numerical data. The OneHotEncoder is designed to encode non-numerical "
                    r"values into numerical values"
                ),
                category=UserWarning,
            )
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
            (
                Table({"a": ["a", "b", "c", "c"], "b": ["a", float("nan"), float("nan"), "a"]}),
                None,
                Table(
                    {
                        "a__a": [1.0, 0.0, 0.0, 0.0],
                        "a__b": [0.0, 1.0, 0.0, 0.0],
                        "a__c": [0.0, 0.0, 1.0, 1.0],
                        "b__a": [1.0, 0.0, 0.0, 1.0],
                        "b__nan": [0.0, 1.0, 1.0, 0.0],
                    },
                ),
            ),
        ],
        ids=[
            "all columns",
            "one column",
            "multiple columns",
            "single underscore counterexample",
            "name conflict",
            "column with nans",
        ],
    )
    def test_should_return_fitted_transformer_and_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        fitted_transformer, transformed_table = OneHotEncoder().fit_and_transform(table, column_names)
        assert fitted_transformer.is_fitted
        assert transformed_table == expected

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
            {"a__b": ["c", "d"], "a": ["b__c", "d"], "b": ["a", float("nan")]},
        )
        added_columns = ["a__b__c", "a__b__d", "a__b__c#2", "a__d", "b__a", "b__nan"]

        transformer = transformer.fit(table, None)
        assert transformer.get_names_of_added_columns() == added_columns

    def test_get_names_of_changed_columns(self) -> None:
        transformer = OneHotEncoder()
        with pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_changed_columns()

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
            (Table({"a": ["a", "b", "b", float("nan")]}), ["a"], Table({"a": ["a", "b", "b", float("nan")]})),
        ],
        ids=[
            "same table to fit and transform",
            "different tables to fit and transform",
            "one column name is a prefix of another column name",
            "column of table contains nan",
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
                "col1": ["a", "b", "b", "c", float("nan")],
            },
        )

        transformer = OneHotEncoder().fit(table, None)
        transformed_table = transformer.transform(table)
        transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1__a": [1.0, 0.0, 0.0, 0.0, 0.0],
                "col1__b": [0.0, 1.0, 1.0, 0.0, 0.0],
                "col1__c": [0.0, 0.0, 0.0, 1.0, 0.0],
                "col1__nan": [0.0, 0.0, 0.0, 0.0, 1.0],
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

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.inverse_transform(table)

    def test_should_raise_if_column_not_found(self) -> None:
        with pytest.raises(ColumnNotFoundError, match=r"Could not find column\(s\) 'col1__one, col1__two'"):
            OneHotEncoder().fit(Table({"col1": ["one", "two"]}), ["col1"]).inverse_transform(
                Table({"col1": [1.0, 0.0]}),
            )

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(
            NonNumericColumnError,
            match=(
                r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1__one',"
                r" 'col1__two'\]"
            ),
        ):
            OneHotEncoder().fit(Table({"col1": ["one", "two"]}), ["col1"]).inverse_transform(
                Table({"col1__one": ["1", "null"], "col1__two": ["2", "ok"]}),
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The OneHotEncoder cannot inverse transform the table because it contains 0 rows",
        ):
            OneHotEncoder().fit(Table({"col1": ["one"]}), ["col1"]).inverse_transform(Table({"col1__one": []}))

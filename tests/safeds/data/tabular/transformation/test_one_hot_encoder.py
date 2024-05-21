import math
import warnings

import pytest
from polars.testing import assert_frame_equal
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import (
    ColumnNotFoundError,
    ColumnTypeError,
    TransformerNotFittedError,
)


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "c"],
            },
        )

        with pytest.raises(ColumnNotFoundError):
            OneHotEncoder(column_names=["col2", "col3"]).fit(table)

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(ValueError, match=r"The OneHotEncoder cannot be fitted because the table contains 0 rows"):
            OneHotEncoder(column_names="col1").fit(Table({"col1": []}))

    def test_should_warn_if_table_contains_numerical_data(self) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                r"The columns \['col1'\] contain numerical data. The OneHotEncoder is designed to encode non-numerical"
                r" values into numerical values"
            ),
        ):
            OneHotEncoder(column_names="col1").fit(Table({"col1": [1, 2, 3]}))

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
                    "col1": [1, 2, math.nan],
                },
            ),
        ],
        ids=["string table", "table with nan"],
    )
    def test_should_not_change_original_transformer(self, table: Table) -> None:
        transformer = OneHotEncoder()
        transformer.fit(table)

        assert transformer._column_names is None
        assert transformer._new_column_names is None
        assert transformer._mapping is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table(
            {
                "col1": ["a", "b", "c"],
                "col2": ["a2", "b2", "c2"],
            },
        )

        transformer = OneHotEncoder().fit(table_to_fit)

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

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)


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
            fitted_transformer = transformer.fit(table)
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
                        "col2": ["a", "b", "b", "c"],
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
                Table({"a": ["a", "b", "c", "c"], "b": [1, math.nan, math.nan, 1]}),
                ["a", "b"],
                Table(
                    {
                        "a__a": [1.0, 0.0, 0.0, 0.0],
                        "a__b": [0.0, 1.0, 0.0, 0.0],
                        "a__c": [0.0, 0.0, 1.0, 1.0],
                        "b__1.0": [1.0, 0.0, 0.0, 1.0],
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
        fitted_transformer, transformed_table = OneHotEncoder(column_names=column_names).fit_and_transform(table)
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
            (Table({"a": [1, 2, 2, float("nan")]}), ["a"], Table({"a": [1, 2, 2, float("nan")]})),
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
        transformer = OneHotEncoder(column_names=column_names).fit(table_to_fit)

        result = transformer.inverse_transform(transformer.transform(table_to_transform))

        # We don't guarantee the order of the columns
        assert set(result.column_names) == set(table_to_transform.column_names)
        assert_frame_equal(
            result._data_frame.select(table_to_transform.column_names),
            table_to_transform._data_frame,
        )

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": ["a", "b", "b", "c"],
            },
        )

        transformer = OneHotEncoder().fit(table)
        transformed_table = transformer.transform(table)
        transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1__a": [1, 0, 0, 0],
                "col1__b": [0, 1, 1, 0],
                "col1__c": [0, 0, 0, 1],
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
        with pytest.raises(ColumnNotFoundError):
            OneHotEncoder(column_names="col1").fit(Table({"col1": ["one", "two"]})).inverse_transform(
                Table({"col1": [1.0, 0.0]}),
            )

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(ColumnTypeError):
            OneHotEncoder(column_names="col1").fit(Table({"col1": ["one", "two"]})).inverse_transform(
                Table({"col1__one": ["1", "null"], "col1__two": ["2", "ok"]}),
            )

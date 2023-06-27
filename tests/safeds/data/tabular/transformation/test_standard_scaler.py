import pytest
import sklearn.exceptions as sk_exceptions
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import StandardScaler
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, UnknownColumnNameError

from tests.helpers import assert_that_tables_are_close


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'col2, col3'"):
            StandardScaler().fit(table, ["col2", "col3"])

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(
            NonNumericColumnError,
            match=r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1', 'col2'\]",
        ):
            StandardScaler().fit(
                Table({"col1": ["one", "two", "apple"], "col2": ["three", "four", "banana"]}),
                ["col1", "col2"],
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(sk_exceptions.NotFittedError,
                           match=r"The StandardScaler cannot be fitted because the table contains 0 rows"):
            StandardScaler().fit(Table({"col1": []}), ["col1"])

    def test_should_not_change_original_transformer(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()
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

        transformer = StandardScaler().fit(table_to_fit, None)

        table_to_transform = Table(
            {
                "col3": ["a", "b", "c"],
            },
        )

        with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'col1, col2'"):
            transformer.transform(table_to_transform)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.transform(table)

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(
            NonNumericColumnError,
            match=r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1', 'col2'\]",
        ):
            StandardScaler().fit(Table({"col1": [1, 2, 3], "col2": [2, 3, 4]}), ["col1", "col2"]).transform(
                Table({"col1": ["a", "b", "c"], "col2": ["b", "c", "e"]}),
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The StandardScaler cannot transform the table because it contains 0 rows",
        ):
            StandardScaler().fit(Table({"col1": [1, 2, 3]}), ["col1"]).transform(Table({"col1": []}))


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = StandardScaler()
        assert not transformer.is_fitted()

    def test_should_return_true_after_fitting(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()
        fitted_transformer = transformer.fit(table, None)
        assert fitted_transformer.is_fitted()


class TestFitAndTransformOnMultipleTables:
    @pytest.mark.parametrize(
        ("fit_and_transform_table", "only_transform_table", "column_names", "expected_1", "expected_2"),
        [
            (
                Table(
                    {
                        "col1": [0.0, 0.0, 1.0, 1.0],
                        "col2": [0.0, 0.0, 1.0, 1.0],
                    },
                ),
                Table(
                    {
                        "col1": [2],
                        "col2": [2],
                    },
                ),
                None,
                Table(
                    {
                        "col1": [-1.0, -1.0, 1.0, 1.0],
                        "col2": [-1.0, -1.0, 1.0, 1.0],
                    },
                ),
                Table(
                    {
                        "col1": [3.0],
                        "col2": [3.0],
                    },
                ),
            ),
        ],
    )
    def test_should_return_transformed_tables(
        self,
        fit_and_transform_table: Table,
        only_transform_table: Table,
        column_names: list[str] | None,
        expected_1: Table,
        expected_2: Table,
    ) -> None:
        s = StandardScaler().fit(fit_and_transform_table, column_names)
        assert s.fit_and_transform(fit_and_transform_table, column_names) == expected_1
        assert s.transform(only_transform_table) == expected_2


class TestFitAndTransform:
    def test_should_not_change_original_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        StandardScaler().fit_and_transform(table)

        expected = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        assert table == expected

    def test_get_names_of_added_columns(self) -> None:
        transformer = StandardScaler()
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
        transformer = StandardScaler()
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
        transformer = StandardScaler()
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
        transformer = StandardScaler().fit(table, None)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        transformer = StandardScaler().fit(table, None)
        transformed_table = transformer.transform(table)
        transformed_table = transformer.inverse_transform(transformed_table)

        expected = Table(
            {
                "col1": [0.0, 0.5, 1.0],
            },
        )

        assert_that_tables_are_close(transformed_table, expected)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()

        with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
            transformer.inverse_transform(table)

    def test_should_raise_if_column_not_found(self) -> None:
        with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'col1, col2'"):
            StandardScaler().fit(Table({"col1": [1, 2, 4], "col2": [2, 3, 4]}), ["col1", "col2"]).inverse_transform(
                Table({"col3": [0, 1, 2]}),
            )

    def test_should_raise_if_table_contains_non_numerical_data(self) -> None:
        with pytest.raises(
            NonNumericColumnError,
            match=r"Tried to do a numerical operation on one or multiple non-numerical columns: \n\['col1', 'col2'\]",
        ):
            StandardScaler().fit(Table({"col1": [1, 2, 4], "col2": [2, 3, 4]}), ["col1", "col2"]).inverse_transform(
                Table({"col1": ["one", "two", "apple"], "col2": ["three", "four", "banana"]}),
            )

    def test_should_raise_if_table_contains_no_rows(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"The StandardScaler cannot transform the table because it contains 0 rows",
        ):
            StandardScaler().fit(Table({"col1": [1, 2, 4]}), ["col1"]).inverse_transform(Table({"col1": []}))

import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import StandardScaler
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table(
            {
                "col1": [0.0, 5.0, 10.0],
            },
        )

        with pytest.raises(UnknownColumnNameError):
            StandardScaler().fit(table, ["col2"])

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
            },
        )

        transformer = StandardScaler().fit(table_to_fit, None)

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
                "col1": [0.0, 5.0, 10.0],
            },
        )

        transformer = StandardScaler()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)


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
                        "col1": [2, 2],
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
                        "col1": [3, 3],
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
        s = StandardScaler().fit(fit_and_transform_table)
        assert s.fit_and_transform(fit_and_transform_table, column_names) == expected_1
        assert s.fit_and_transform(only_transform_table, column_names) == expected_2


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
        with pytest.warns(
            UserWarning,
            match="StandardScaler only changes data within columns, but does not add any columns.",
        ), pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_added_columns()

        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        with pytest.warns(
            UserWarning,
            match="StandardScaler only changes data within columns, but does not add any columns.",
        ):
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
        with pytest.warns(
            UserWarning,
            match="StandardScaler only changes data within columns, but does not remove any columns.",
        ), pytest.raises(TransformerNotFittedError):
            transformer.get_names_of_removed_columns()

        table = Table(
            {
                "a": [0.0],
            },
        )
        transformer = transformer.fit(table, None)
        with pytest.warns(
            UserWarning,
            match="StandardScaler only changes data within columns, but does not remove any columns.",
        ):
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

        transformer = StandardScaler()

        with pytest.raises(TransformerNotFittedError):
            transformer.inverse_transform(table)

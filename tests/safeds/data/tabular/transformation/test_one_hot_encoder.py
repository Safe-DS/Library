import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import TransformerNotFittedError, UnknownColumnNameError
from safeds.data.tabular.transformation import OneHotEncoder


class TestFit:
    def test_should_raise_if_column_not_found(self) -> None:
        table = Table.from_columns(
            [
                Column("col1", ["a", "b", "c"]),
            ],
        )

        with pytest.raises(UnknownColumnNameError):
            OneHotEncoder().fit(table, ["col2"])

    def test_should_not_change_original_transformer(self) -> None:
        table = Table.from_columns(
            [
                Column("col1", ["a", "b", "c"]),
            ],
        )

        transformer = OneHotEncoder()
        transformer.fit(table)

        assert transformer._wrapped_transformer is None
        assert transformer._column_names is None


class TestTransform:
    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table.from_columns(
            [
                Column("col1", ["a", "b", "c"]),
            ],
        )

        transformer = OneHotEncoder().fit(table_to_fit)

        table_to_transform = Table.from_columns(
            [
                Column("col2", ["a", "b", "c"]),
            ],
        )

        with pytest.raises(UnknownColumnNameError):
            transformer.transform(table_to_transform)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table.from_columns(
            [
                Column("col1", ["a", "b", "c"]),
            ],
        )

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError):
            transformer.transform(table)


class TestIsFitted:
    def test_should_return_false_before_fitting(self) -> None:
        transformer = OneHotEncoder()
        assert not transformer.is_fitted()

    def test_should_return_true_after_fitting(self) -> None:
        table = Table.from_columns(
            [
                Column("col1", ["a", "b", "c"]),
            ],
        )

        transformer = OneHotEncoder()
        fitted_transformer = transformer.fit(table)
        assert fitted_transformer.is_fitted()


class TestFitAndTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table.from_columns(
                    [
                        Column("col1", ["a", "b", "b", "c"]),
                    ],
                ),
                None,
                Table.from_columns(
                    [
                        Column("col1_a", [1.0, 0.0, 0.0, 0.0]),
                        Column("col1_b", [0.0, 1.0, 1.0, 0.0]),
                        Column("col1_c", [0.0, 0.0, 0.0, 1.0]),
                    ],
                ),
            ),
            (
                Table.from_columns(
                    [
                        Column("col1", ["a", "b", "b", "c"]),
                        Column("col2", ["a", "b", "b", "c"]),
                    ],
                ),
                ["col1"],
                Table.from_columns(
                    [
                        Column("col2", ["a", "b", "b", "c"]),
                        Column("col1_a", [1.0, 0.0, 0.0, 0.0]),
                        Column("col1_b", [0.0, 1.0, 1.0, 0.0]),
                        Column("col1_c", [0.0, 0.0, 0.0, 1.0]),
                    ],
                ),
            ),
        ],
    )
    def test_should_return_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        assert OneHotEncoder().fit_and_transform(table, column_names) == expected

    def test_should_not_change_original_table(self) -> None:
        table = Table.from_columns(
            [
                Column("col1", ["a", "b", "c"]),
            ],
        )

        OneHotEncoder().fit_and_transform(table)

        expected = Table.from_columns(
            [
                Column("col1", ["a", "b", "c"]),
            ],
        )

        assert table == expected


class TestInverseTransform:
    @pytest.mark.parametrize(
        "table",
        [
            Table.from_columns(
                [
                    Column("col1", ["a", "b", "b", "c"]),
                ],
            ),
        ],
    )
    def test_should_return_original_table(self, table: Table) -> None:
        transformer = OneHotEncoder().fit(table)

        assert transformer.inverse_transform(transformer.transform(table)) == table

    def test_should_not_change_transformed_table(self) -> None:
        table = Table.from_columns(
            [
                Column("col1", ["a", "b", "b", "c"]),
            ],
        )

        transformer = OneHotEncoder().fit(table)
        transformed_table = transformer.transform(table)
        transformer.inverse_transform(transformed_table)

        expected = Table.from_columns(
            [
                Column("col1_a", [1.0, 0.0, 0.0, 0.0]),
                Column("col1_b", [0.0, 1.0, 1.0, 0.0]),
                Column("col1_c", [0.0, 0.0, 0.0, 1.0]),
            ],
        )

        assert transformed_table == expected

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table.from_columns(
            [
                Column("a", [1.0, 0.0, 0.0, 0.0]),
                Column("b", [0.0, 1.0, 1.0, 0.0]),
                Column("c", [0.0, 0.0, 0.0, 1.0]),
            ],
        )

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError):
            transformer.inverse_transform(table)

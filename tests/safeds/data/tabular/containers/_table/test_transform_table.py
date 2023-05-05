import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import TransformerNotFittedError, UnknownColumnNameError
from safeds.data.tabular.transformation import OneHotEncoder


class TestTransform:
    @pytest.mark.parametrize(
        ("table", "column_names", "expected"),
        [
            (
                Table.from_dict(
                    {
                        "col1": ["a", "b", "b", "c"],
                    },
                ),
                None,
                Table.from_dict(
                    {
                        "col1_a": [1.0, 0.0, 0.0, 0.0],
                        "col1_b": [0.0, 1.0, 1.0, 0.0],
                        "col1_c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
            ),
            (
                Table.from_dict(
                    {
                        "col1": ["a", "b", "b", "c"],
                        "col2": ["a", "b", "b", "c"],
                    },
                ),
                ["col1"],
                Table.from_dict(
                    {
                        "col1_a": [1.0, 0.0, 0.0, 0.0],
                        "col1_b": [0.0, 1.0, 1.0, 0.0],
                        "col1_c": [0.0, 0.0, 0.0, 1.0],
                        "col2": ["a", "b", "b", "c"],
                    },
                ),
            ),
            (
                Table.from_dict(
                    {
                        "col1": ["a", "b", "b", "c"],
                        "col2": ["a", "b", "b", "c"],
                    },
                ),
                ["col1", "col2"],
                Table.from_dict(
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
            (
                Table.from_dict(
                    {
                        "col1": ["a", "b", "c"],
                    },
                ),
                [],
                Table.from_dict(
                    {
                        "col1": ["a", "b", "c"],
                    },
                ),
            ),
        ],
        ids=["all columns", "one column", "multiple columns", "none"],
    )
    def test_should_return_transformed_table(
        self,
        table: Table,
        column_names: list[str] | None,
        expected: Table,
    ) -> None:
        transformer = OneHotEncoder().fit(table, column_names)
        assert table.transform_table(transformer) == expected

    def test_should_raise_if_column_not_found(self) -> None:
        table_to_fit = Table.from_dict(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = OneHotEncoder().fit(table_to_fit, None)

        table_to_transform = Table.from_dict(
            {
                "col2": ["a", "b", "c"],
            },
        )

        with pytest.raises(UnknownColumnNameError):
            table_to_transform.transform_table(transformer)

    def test_should_raise_if_not_fitted(self) -> None:
        table = Table.from_dict(
            {
                "col1": ["a", "b", "c"],
            },
        )

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError):
            table.transform_table(transformer)

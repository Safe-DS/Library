import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import TransformerNotFittedError
from safeds.data.tabular.transformation import OneHotEncoder


class TestInverseTransformTable:
    @pytest.mark.parametrize(
        ("table_to_fit", "column_names", "table_to_transform"),
        [
            (
                Table.from_dict(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
                ["b"],
                Table.from_dict(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
            ),
            (
                Table.from_dict(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "c": [0.0, 0.0, 0.0, 1.0],
                    },
                ),
                ["b"],
                Table.from_dict(
                    {
                        "c": [0.0, 0.0, 0.0, 1.0],
                        "b": ["a", "b", "b", "c"],
                        "a": [1.0, 0.0, 0.0, 0.0],
                    },
                ),
            ),
            (
                Table.from_dict(
                    {
                        "a": [1.0, 0.0, 0.0, 0.0],
                        "b": ["a", "b", "b", "c"],
                        "bb": ["a", "b", "b", "c"],
                    },
                ),
                ["b", "bb"],
                Table.from_dict(
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
        transformed_table = transformer.transform(table_to_transform)

        result = transformed_table.inverse_transform_table(transformer)

        # This checks whether the columns are in the same order
        assert result.column_names == table_to_transform.column_names
        # This is subsumed by the next assertion, but we get a better error message
        assert result.schema == table_to_transform.schema
        assert result == table_to_transform

    def test_should_not_change_transformed_table(self) -> None:
        table = Table.from_dict(
            {
                "col1": ["a", "b", "b", "c"],
            },
        )

        transformer = OneHotEncoder().fit(table, None)
        transformed_table = transformer.transform(table)
        transformed_table.inverse_transform_table(transformer)

        expected = Table.from_dict(
            {
                "col1_a": [1.0, 0.0, 0.0, 0.0],
                "col1_b": [0.0, 1.0, 1.0, 0.0],
                "col1_c": [0.0, 0.0, 0.0, 1.0],
            },
        )

        assert transformed_table == expected

    def test_should_raise_error_if_not_fitted(self) -> None:
        table = Table.from_dict(
            {
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.0, 1.0, 1.0, 0.0],
                "c": [0.0, 0.0, 0.0, 1.0],
            },
        )

        transformer = OneHotEncoder()

        with pytest.raises(TransformerNotFittedError):
            table.inverse_transform_table(transformer)

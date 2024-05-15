import pytest
from polars.testing import assert_frame_equal
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import TransformerNotFittedError


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
    table_to_fit: Table,
    column_names: list[str],
    table_to_transform: Table,
) -> None:
    transformer = OneHotEncoder().fit(table_to_fit, column_names)
    transformed_table = transformer.transform(table_to_transform)

    result = transformed_table.inverse_transform_table(transformer)

    # Order is not guaranteed
    assert set(result.column_names) == set(table_to_transform.column_names)
    assert_frame_equal(
        result._data_frame.select(table_to_transform.column_names),
        table_to_transform._data_frame,
    )


def test_should_not_change_transformed_table() -> None:
    table = Table(
        {
            "col1": ["a", "b", "b", "c"],
        },
    )

    transformer = OneHotEncoder().fit(table, None)
    transformed_table = transformer.transform(table)
    transformed_table.inverse_transform_table(transformer)

    expected = Table(
        {
            "col1__a": [1.0, 0.0, 0.0, 0.0],
            "col1__b": [0.0, 1.0, 1.0, 0.0],
            "col1__c": [0.0, 0.0, 0.0, 1.0],
        },
    )

    assert transformed_table == expected


def test_should_raise_error_if_not_fitted() -> None:
    table = Table(
        {
            "a": [1.0, 0.0, 0.0, 0.0],
            "b": [0.0, 1.0, 1.0, 0.0],
            "c": [0.0, 0.0, 0.0, 1.0],
        },
    )

    transformer = OneHotEncoder()

    with pytest.raises(TransformerNotFittedError, match=r"The transformer has not been fitted yet."):
        table.inverse_transform_table(transformer)

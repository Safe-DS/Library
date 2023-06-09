import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import TransformerNotFittedError, UnknownColumnNameError


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
                    "col1": ["a", "b", "c"],
                },
            ),
            [],
            Table(
                {
                    "col1": ["a", "b", "c"],
                },
            ),
        ),
    ],
    ids=["all columns", "one column", "multiple columns", "none"],
)
def test_should_return_transformed_table(
    table: Table,
    column_names: list[str] | None,
    expected: Table,
) -> None:
    transformer = OneHotEncoder().fit(table, column_names)
    assert table.transform_table(transformer) == expected


def test_should_raise_if_column_not_found() -> None:
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
        table_to_transform.transform_table(transformer)


def test_should_raise_if_not_fitted() -> None:
    table = Table(
        {
            "col1": ["a", "b", "c"],
        },
    )

    transformer = OneHotEncoder()

    with pytest.raises(TransformerNotFittedError):
        table.transform_table(transformer)

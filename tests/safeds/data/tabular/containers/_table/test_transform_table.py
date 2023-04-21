import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError, TransformerNotFittedError
from safeds.data.tabular.transformation import OneHotEncoder


class TestTransform:
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


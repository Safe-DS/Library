import sys

import pytest

from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    "schema",
    [
        Schema({}),
        Schema({"col1": ColumnType.null()}),
        Schema({"col1": ColumnType.null(), "col2": ColumnType.null()}),
    ],
    ids=[
        "empty",
        "one column",
        "two columns",
    ],
)
def test_should_size_be_greater_than_normal_object(schema: Schema) -> None:
    assert sys.getsizeof(schema) > sys.getsizeof(object())

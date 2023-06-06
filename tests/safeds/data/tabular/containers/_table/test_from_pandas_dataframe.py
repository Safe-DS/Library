import pandas as pd
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import Integer, RealNumber, Schema, String


@pytest.mark.parametrize(
    ("dataframe", "schema", "expected", "expected_table"),
    [
        (pd.DataFrame({"col1": [0]}), Schema({"col1": Integer()}), Schema({"col1": Integer()}), Table({"col1": [0]})),
        (
            pd.DataFrame({"col1": [0], "col2": ["a"]}),
            Schema({"col1": Integer(), "col2": String()}),
            Schema({"col1": Integer(), "col2": String()}),
            Table({"col1": [0], "col2": ["a"]}),
        ),
        (
            pd.DataFrame({"col1": [0, 1.1]}),
            Schema({"col1": String()}),
            Schema({"col1": String()}),
            Table({"col1": [0, 1.1]}),
        ),
        (
            pd.DataFrame({"col1": [0, 1.1], "col2": ["a", "b"]}),
            Schema({"col1": String(), "col2": String()}),
            Schema({"col1": String(), "col2": String()}),
            Table({"col1": [0, 1.1], "col2": ["a", "b"]}),
        ),
        (pd.DataFrame(), Schema({}), Schema({}), Table()),
    ],
    ids=["one row, one column", "one row, two columns", "two rows, one column", "two rows, two columns", "empty"],
)
def test_should_use_the_schema_if_passed(
    dataframe: pd.DataFrame, schema: Schema, expected: Schema, expected_table: Table,
) -> None:
    table = Table._from_pandas_dataframe(dataframe, schema)
    assert table._schema == expected
    assert table == expected_table


@pytest.mark.parametrize(
    ("dataframe", "expected"),
    [
        (
            pd.DataFrame({"col1": [0]}),
            Schema({"col1": Integer()}),
        ),
        (
            pd.DataFrame({"col1": [0], "col2": ["a"]}),
            Schema({"col1": Integer(), "col2": String()}),
        ),
        (
            pd.DataFrame({"col1": [0, 1.1]}),
            Schema({"col1": RealNumber()}),
        ),
        (
            pd.DataFrame({"col1": [0, 1.1], "col2": ["a", "b"]}),
            Schema({"col1": RealNumber(), "col2": String()}),
        ),
    ],
    ids=[
        "one row, one column",
        "one row, two columns",
        "two rows, one column",
        "two rows, two columns",
    ],
)
def test_should_infer_the_schema_if_not_passed(dataframe: pd.DataFrame, expected: Schema) -> None:
    table = Table._from_pandas_dataframe(dataframe)
    assert table._schema == expected


def test_should_be_able_to_handle_empty_dataframe_with_given_schema() -> None:
    table = Table._from_pandas_dataframe(pd.DataFrame(), Schema({"col1": Integer(), "col2": Integer()}))
    table.get_column("col1")

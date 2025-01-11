from typing import Any

import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("schema_1", "schema_2", "expected"),
    [
        # equal (empty)
        (
            Schema({}),
            Schema({}),
            True,
        ),
        # equal (one column)
        (
            Schema({"col1": ColumnType.null()}),
            Schema({"col1": ColumnType.null()}),
            True,
        ),
        # equal (two columns)
        (
            Schema({"col1": ColumnType.null(), "col2": ColumnType.int8()}),
            Schema({"col1": ColumnType.null(), "col2": ColumnType.int8()}),
            True,
        ),
        # not equal (too few columns)
        (
            Schema({"col1": ColumnType.null()}),
            Schema({}),
            False,
        ),
        # not equal (too many columns)
        (
            Schema({}),
            Schema({"col1": ColumnType.null()}),
            False,
        ),
        # not equal (different column order)
        (
            Schema({"col1": ColumnType.null(), "col2": ColumnType.int8()}),
            Schema({"col2": ColumnType.int8(), "col1": ColumnType.null()}),
            False,
        ),
        # not equal (different column names)
        (
            Schema({"col1": ColumnType.null()}),
            Schema({"col2": ColumnType.null()}),
            False,
        ),
        # not equal (different types)
        (
            Schema({"col1": ColumnType.null()}),
            Schema({"col1": ColumnType.int8()}),
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (empty)",
        "equal (one column)",
        "equal (two columns)",
        # Not equal
        "not equal (too few columns)",
        "not equal (too many columns)",
        "not equal (different column order)",
        "not equal (different column names)",
        "not equal (different types)",
    ],
)
def test_should_return_whether_schemas_are_equal(schema_1: Schema, schema_2: Schema, expected: bool) -> None:
    assert (schema_1.__eq__(schema_2)) == expected


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
def test_should_return_true_if_schemas_are_identical(schema: Schema) -> None:
    assert (schema.__eq__(schema)) is True


@pytest.mark.parametrize(
    ("schema", "other"),
    [
        (Schema({}), None),
        (Schema({}), Column("col1", [])),
    ],
    ids=[
        "Schema vs. None",
        "Schema vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_is_not_schema(schema: Schema, other: Any) -> None:
    assert (schema.__eq__(other)) is NotImplemented

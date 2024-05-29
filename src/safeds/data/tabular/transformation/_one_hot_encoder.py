from __future__ import annotations

import warnings
from typing import Any

from safeds._utils import _structural_hash
from safeds._validation import _check_columns_exist
from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    TransformerNotFittedError,
)

from ._invertible_table_transformer import InvertibleTableTransformer


class OneHotEncoder(InvertibleTableTransformer):
    """
    A way to deal with categorical features that is particularly useful for unordered (i.e. nominal) data.

    It replaces a column with a set of columns, each representing a unique value in the original column. The value of
    each new column is 1 if the original column had that value, and 0 otherwise. Take the following table as an
    example:

    | col1 |
    |------|
    | "a"  |
    | "b"  |
    | "c"  |
    | "a"  |

    The one-hot encoding of this table is:

    | col1__a | col1__b | col1__c |
    |---------|---------|---------|
    | 1       | 0       | 0       |
    | 0       | 1       | 0       |
    | 0       | 0       | 1       |
    | 1       | 0       | 0       |

    The name "one-hot" comes from the fact that each row has exactly one 1 in it, and the rest of the values are 0s.
    One-hot encoding is closely related to dummy variable / indicator variables, which are used in statistics.

    Parameters
    ----------
    column_names:
        The list of columns used to fit the transformer. If `None`, all non-numeric columns are used.
    separator:
        The separator used to separate the original column name from the value in the new column names.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Table
    >>> from safeds.data.tabular.transformation import OneHotEncoder
    >>> table = Table({"col1": ["a", "b", "c", "a"]})
    >>> transformer = OneHotEncoder()
    >>> transformer.fit_and_transform(table)[1]
    +---------+---------+---------+
    | col1__a | col1__b | col1__c |
    |     --- |     --- |     --- |
    |      u8 |      u8 |      u8 |
    +=============================+
    |       1 |       0 |       0 |
    |       0 |       1 |       0 |
    |       0 |       0 |       1 |
    |       1 |       0 |       0 |
    +---------+---------+---------+
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        column_names: str | list[str] | None = None,
        separator: str = "__",
    ) -> None:
        super().__init__(column_names)

        # Parameters
        self._separator = separator

        # Internal state
        self._new_column_names: list[str] | None = None
        self._mapping: dict[str, list[tuple[str, Any]]] | None = None  # Column name -> (new column name, value)[]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OneHotEncoder):
            return NotImplemented
        elif self is other:
            return True

        return self._separator == other._separator and self._mapping == other._mapping

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._separator,
            # TODO: Leave out the internal state for faster hashing
            self._mapping,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._mapping is not None

    @property
    def separator(self) -> str:
        """The separator used to separate the original column name from the value in the new column names."""
        return self._separator

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> OneHotEncoder:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer.

        Returns
        -------
        fitted_transformer:
            The fitted transformer.

        Raises
        ------
        ColumnNotFoundError
            If column_names contain a column name that is missing in the table.
        ValueError
            If the table contains 0 rows.
        """
        if self._column_names is None:
            column_names = [name for name in table.column_names if not table.get_column_type(name).is_numeric]
        else:
            column_names = self._column_names
            _check_columns_exist(table, column_names)
            _warn_if_columns_are_numeric(table, column_names)

        if table.row_count == 0:
            raise ValueError("The OneHotEncoder cannot be fitted because the table contains 0 rows")

        # Learn the transformation
        new_column_names: list[str] = []
        mapping: dict[str, list[tuple[str, Any]]] = {}

        known_names = set(table.column_names)

        for name in column_names:
            mapping[name] = []
            for value in table.get_column(name).get_distinct_values():
                base_name = f"{name}{self._separator}{value}"
                new_name = base_name

                # Ensure that the new column name is unique
                counter = 2
                while new_name in known_names:
                    new_name = f"{base_name}#{counter}"
                    counter += 1

                known_names.add(new_name)
                new_column_names.append(new_name)
                mapping[name].append((new_name, value))

        # Create a copy with the learned transformation
        result = OneHotEncoder(column_names=column_names, separator=self._separator)
        result._new_column_names = new_column_names
        result._mapping = mapping

        return result

    def transform(self, table: Table) -> Table:
        """
        Apply the learned transformation to a table.

        The table is not modified.

        Parameters
        ----------
        table:
            The table to which the learned transformation is applied.

        Returns
        -------
        transformed_table:
            The transformed table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        ColumnNotFoundError
            If the input table does not contain all columns used to fit the transformer.
        """
        import polars as pl

        # Used in favor of is_fitted, so the type checker is happy
        if self._column_names is None or self._mapping is None:
            raise TransformerNotFittedError

        _check_columns_exist(table, self._column_names)

        expressions = [
            # UInt8 can be used without conversion in scikit-learn
            pl.col(column_name).eq_missing(value).alias(new_name).cast(pl.UInt8)
            for column_name in self._column_names
            for new_name, value in self._mapping[column_name]
        ]

        return Table._from_polars_lazy_frame(
            table._lazy_frame.with_columns(expressions).drop(self._column_names),
        )

    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Undo the learned transformation.

        The table is not modified.

        Parameters
        ----------
        transformed_table:
            The table to be transformed back to the original version.

        Returns
        -------
        table:
            The original table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        ColumnNotFoundError
            If the input table does not contain all columns used to fit the transformer.
        NonNumericColumnError
            If the transformed columns of the input table contain non-numerical data.
        """
        import polars as pl

        # Used in favor of is_fitted, so the type checker is happy
        if self._column_names is None or self._new_column_names is None or self._mapping is None:
            raise TransformerNotFittedError

        _check_columns_exist(transformed_table, self._new_column_names)
        _check_columns_are_numeric(
            transformed_table,
            self._new_column_names,
            operation="inverse-transform with a OneHotEncoder",
        )

        expressions = [
            pl.coalesce(
                [
                    # The pl.lit is needed, so strings are not interpreted as column names
                    pl.when(pl.col(new_column_name) == 1).then(pl.lit(value))
                    for new_column_name, value in self._mapping[column_name]
                ],
            ).alias(column_name)
            for column_name in self._mapping
        ]

        return Table._from_polars_lazy_frame(
            transformed_table._lazy_frame.with_columns(expressions).drop(self._new_column_names),
        )

    # TODO: remove / replace with consistent introspection methods across all transformers
    def _get_names_of_added_columns(self) -> list[str]:
        if self._new_column_names is None:
            raise TransformerNotFittedError
        return list(self._new_column_names)  # defensive copy


def _warn_if_columns_are_numeric(table: Table, column_names: list[str]) -> None:
    numeric_columns = table.remove_columns_except(column_names).remove_non_numeric_columns().column_names
    if numeric_columns:
        warnings.warn(
            f"The columns {numeric_columns} contain numerical data. "
            "The OneHotEncoder is designed to encode non-numerical values into numerical values",
            UserWarning,
            stacklevel=2,
        )

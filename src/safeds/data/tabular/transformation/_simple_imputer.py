from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

from safeds._utils import _structural_hash
from safeds._validation import _check_columns_exist
from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.tabular.containers import Table
from safeds.exceptions import TransformerNotFittedError

from ._table_transformer import TableTransformer


class SimpleImputer(TableTransformer):
    """
    Replace missing values using the given strategy.

    Parameters
    ----------
    strategy:
        How to replace missing values.
    value_to_replace:
        The value that should be replaced.
    column_names:
        The list of columns used to fit the transformer. If `None`, all columns are used.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column, Table
    >>> from safeds.data.tabular.transformation import SimpleImputer
    >>>
    >>> table = Table.from_columns(
    ...     [
    ...         Column("a", [1, 3, None]),
    ...         Column("b", [None, 2, 3]),
    ...     ],
    ... )
    >>> transformer = SimpleImputer(SimpleImputer.Strategy.constant(0))
    >>> transformed_table = transformer.fit_and_transform(table)
    """

    class Strategy(ABC):
        """
        Various strategies to replace missing values.

        Use the static factory methods to create instances of this class.
        """

        @abstractmethod
        def __eq__(self, other: object) -> bool: ...

        @abstractmethod
        def __hash__(self) -> int: ...

        @abstractmethod
        def __str__(self) -> str: ...

        @abstractmethod
        def _get_replacement(self, table: Table) -> dict[str, Any]:
            """Return a polars expression to compute the replacement value for each column of a data frame."""

        @staticmethod
        def constant(value: Any) -> SimpleImputer.Strategy:
            """
            Replace missing values with the given constant value.

            Parameters
            ----------
            value:
                The value to replace missing values.
            """
            return _Constant(value)

        @staticmethod
        def mean() -> SimpleImputer.Strategy:
            """Replace missing values with the mean of each column."""
            return _Mean()

        @staticmethod
        def median() -> SimpleImputer.Strategy:
            """Replace missing values with the median of each column."""
            return _Median()

        @staticmethod
        def mode() -> SimpleImputer.Strategy:
            """Replace missing values with the mode of each column."""
            return _Mode()

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        strategy: SimpleImputer.Strategy,
        *,
        column_names: str | list[str] | None = None,
        value_to_replace: float | str | None = None,
    ) -> None:
        super().__init__(column_names)

        # Parameters
        self._strategy = strategy
        self._value_to_replace = value_to_replace

        # Internal state
        self._replacement: dict[str, Any] | None = None

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._strategy,
            self._value_to_replace,
            # Leave out the internal state for faster hashing
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._replacement is not None

    @property
    def strategy(self) -> SimpleImputer.Strategy:
        """The strategy used to replace missing values."""
        return self._strategy

    @property
    def value_to_replace(self) -> Any:
        """The value that should be replaced."""
        return self._value_to_replace

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> SimpleImputer:
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
            If column_names contain a column name that is missing in the table
        ValueError
            If the table contains 0 rows
        NonNumericColumnError
            If the strategy is set to either Mean or Median and the specified columns of the table contain non-numerical
            data.
        """
        if isinstance(self._strategy, _Mean | _Median):
            if self._column_names is None:
                column_names = [name for name in table.column_names if table.get_column_type(name).is_numeric]
            else:
                column_names = self._column_names
                _check_columns_exist(table, column_names)
                _check_columns_are_numeric(table, column_names, operation="fit a SimpleImputer")
        else:  # noqa: PLR5501
            if self._column_names is None:
                column_names = table.column_names
            else:
                column_names = self._column_names
                _check_columns_exist(table, column_names)

        if table.row_count == 0:
            raise ValueError("The SimpleImputer cannot be fitted because the table contains 0 rows")

        # Learn the transformation
        replacement = self._strategy._get_replacement(table)

        # Create a copy with the learned transformation
        result = SimpleImputer(self._strategy, column_names=column_names, value_to_replace=self._value_to_replace)
        result._replacement = replacement

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
        if self._column_names is None or self._replacement is None:
            raise TransformerNotFittedError

        _check_columns_exist(table, self._column_names)

        columns = [
            (pl.col(name).replace(old=self._value_to_replace, new=self._replacement[name]))
            for name in self._column_names
        ]

        return Table._from_polars_lazy_frame(
            table._lazy_frame.with_columns(columns),
        )


# ----------------------------------------------------------------------------------------------------------------------
# Imputation strategies
# ----------------------------------------------------------------------------------------------------------------------


class _Constant(SimpleImputer.Strategy):
    def __init__(self, value: Any):
        self._value = value

    @property
    def value(self) -> Any:
        return self._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Constant):
            return NotImplemented
        if self is other:
            return True
        return self._value == other._value

    def __hash__(self) -> int:
        return _structural_hash(str(self))

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._value)

    def __str__(self) -> str:
        return f"Constant({self._value})"

    def _get_replacement(self, table: Table) -> dict[str, Any]:
        return {name: self._value for name in table.column_names}


class _Mean(SimpleImputer.Strategy):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Mean):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(str(self))

    def __str__(self) -> str:
        return "Mean"

    def _get_replacement(self, table: Table) -> dict[str, Any]:
        return table._lazy_frame.mean().collect().to_dict()


class _Median(SimpleImputer.Strategy):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Median):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(str(self))

    def __str__(self) -> str:
        return "Median"

    def _get_replacement(self, table: Table) -> dict[str, Any]:
        return table._lazy_frame.median().collect().to_dict()


class _Mode(SimpleImputer.Strategy):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Mode):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(str(self))

    def __str__(self) -> str:
        return "Mode"

    def _get_replacement(self, table: Table) -> dict[str, Any]:
        return {name: table.get_column(name).mode()[0] for name in table.column_names}


# Override the methods with classes, so they can be used in `isinstance` calls. Unlike methods, classes define a type.
# This is needed for the DSL, where imputer strategies are variants of an enum.
SimpleImputer.Strategy.constant = _Constant  # type: ignore[method-assign]
SimpleImputer.Strategy.mean = _Mean  # type: ignore[method-assign]
SimpleImputer.Strategy.median = _Median  # type: ignore[method-assign]
SimpleImputer.Strategy.mode = _Mode  # type: ignore[method-assign]

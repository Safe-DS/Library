from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation._table_transformer import TableTransformer
from safeds.exceptions import NonNumericColumnError, TransformerNotFittedError, UnknownColumnNameError

if TYPE_CHECKING:
    from sklearn.impute import SimpleImputer as sk_SimpleImputer


class Imputer(TableTransformer):
    """
    Replace missing values with the given strategy.

    Parameters
    ----------
    strategy:
        The strategy used to impute missing values. Use the classes nested inside `Imputer.Strategy` to specify it.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column, Table
    >>> from safeds.data.tabular.transformation import Imputer
    >>>
    >>> table = Table.from_columns(
    ...     [
    ...         Column("a", [1, 3, None]),
    ...         Column("b", [None, 2, 3]),
    ...     ],
    ... )
    >>> transformer = Imputer(Imputer.Strategy.Constant(0))
    >>> transformed_table = transformer.fit_and_transform(table)
    """

    class Strategy(ABC):
        """
        The abstract base class of the different imputation strategies supported by the `Imputer`.

        This class is only needed for type annotations. Use the subclasses nested inside `Imputer.Strategy` instead.
        """

        @abstractmethod
        def __eq__(self, other: object) -> bool:
            pass

        @abstractmethod
        def __hash__(self) -> int:
            pass

        @abstractmethod
        def _apply(self, imputer: sk_SimpleImputer) -> None:
            """
            Set the imputer strategy of the given imputer.

            Parameters
            ----------
            imputer:
                The imputer to augment.
            """

        @staticmethod
        def Constant(value: Any) -> Imputer.Strategy:  # noqa: N802
            """
            Replace missing values with the given constant value.

            Parameters
            ----------
            value:
                The value to replace missing values.
            """
            return _Constant(value)

        @staticmethod
        def Mean() -> Imputer.Strategy:  # noqa: N802
            """Replace missing values with the mean of each column."""
            return _Mean()

        @staticmethod
        def Median() -> Imputer.Strategy:  # noqa: N802
            """Replace missing values with the median of each column."""
            return _Median()

        @staticmethod
        def Mode() -> Imputer.Strategy:  # noqa: N802
            """Replace missing values with the mode of each column."""
            return _Mode()

    def __init__(self, strategy: Imputer.Strategy):
        self._strategy = strategy

        self._wrapped_transformer: sk_SimpleImputer | None = None
        self._column_names: list[str] | None = None

    # noinspection PyProtectedMember
    def fit(self, table: Table, column_names: list[str] | None) -> Imputer:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer.
        column_names:
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        fitted_transformer:
            The fitted transformer.

        Raises
        ------
        UnknownColumnNameError
            If column_names contain a column name that is missing in the table
        ValueError
            If the table contains 0 rows
        NonNumericColumnError
            If the strategy is set to either Mean or Median and the specified columns of the table contain non-numerical data.
        """
        from sklearn.impute import SimpleImputer as sk_SimpleImputer

        if column_names is None:
            column_names = table.column_names
        else:
            missing_columns = sorted(set(column_names) - set(table.column_names))
            if len(missing_columns) > 0:
                raise UnknownColumnNameError(missing_columns)

        if table.number_of_rows == 0:
            raise ValueError("The Imputer cannot be fitted because the table contains 0 rows")

        if (isinstance(self._strategy, _Mean | _Median)) and table.keep_only_columns(
            column_names,
        ).remove_columns_with_non_numerical_values().number_of_columns < len(
            column_names,
        ):
            raise NonNumericColumnError(
                str(
                    sorted(
                        set(table.keep_only_columns(column_names).column_names)
                        - set(
                            table.keep_only_columns(column_names)
                            .remove_columns_with_non_numerical_values()
                            .column_names,
                        ),
                    ),
                ),
            )

        if isinstance(self._strategy, _Mode):
            multiple_most_frequent = {}
            for name in column_names:
                if len(table.get_column(name).mode()) > 1:
                    multiple_most_frequent[name] = table.get_column(name).mode()
            if len(multiple_most_frequent) > 0:
                warnings.warn(
                    "There are multiple most frequent values in a column given to the Imputer.\nThe lowest values"
                    " are being chosen in this cases. The following columns have multiple most frequent"
                    f" values:\n{multiple_most_frequent}",
                    UserWarning,
                    stacklevel=2,
                )

        wrapped_transformer = sk_SimpleImputer()
        self._strategy._apply(wrapped_transformer)
        wrapped_transformer.fit(table._data[column_names])

        result = Imputer(self._strategy)
        result._wrapped_transformer = wrapped_transformer
        result._column_names = column_names

        return result

    # noinspection PyProtectedMember
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
        UnknownColumnNameError
            If the input table does not contain all columns used to fit the transformer.
        ValueError
            If the table contains 0 rows.
        """
        import pandas as pd

        # Transformer has not been fitted yet
        if self._wrapped_transformer is None or self._column_names is None:
            raise TransformerNotFittedError

        # Input table does not contain all columns used to fit the transformer
        missing_columns = sorted(set(self._column_names) - set(table.column_names))
        if len(missing_columns) > 0:
            raise UnknownColumnNameError(missing_columns)

        if table.number_of_rows == 0:
            raise ValueError("The Imputer cannot transform the table because it contains 0 rows")

        data = table._data.reset_index(drop=True)
        data[self._column_names] = pd.DataFrame(
            self._wrapped_transformer.transform(data[self._column_names]),
            columns=self._column_names,
        )
        return Table._from_pandas_dataframe(data, table.schema)

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._wrapped_transformer is not None

    def get_names_of_added_columns(self) -> list[str]:
        """
        Get the names of all new columns that have been added by the Imputer.

        Returns
        -------
        added_columns:
            A list of names of the added columns, ordered as they will appear in the table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError
        return []

    def get_names_of_changed_columns(self) -> list[str]:
        """
         Get the names of all columns that may have been changed by the Imputer.

        Returns
        -------
        changed_columns:
             The list of (potentially) changed column names, as passed to fit.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError
        return self._column_names

    def get_names_of_removed_columns(self) -> list[str]:
        """
        Get the names of all columns that have been removed by the Imputer.

        Returns
        -------
        removed_columns:
            A list of names of the removed columns, ordered as they appear in the table the Imputer was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError
        return []


# ----------------------------------------------------------------------------------------------------------------------
# Imputation strategies
# ----------------------------------------------------------------------------------------------------------------------

class _Constant(Imputer.Strategy):
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

    def __hash__(self):
        return _structural_hash(str(self))

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._value)

    def __str__(self) -> str:
        return f"Constant({self._value})"

    def _apply(self, imputer: sk_SimpleImputer) -> None:
        imputer.strategy = "constant"
        imputer.fill_value = self._value


class _Mean(Imputer.Strategy):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Mean):
            return NotImplemented
        return True

    def __hash__(self):
        return _structural_hash(str(self))

    def __str__(self) -> str:
        return "Mean"

    def _apply(self, imputer: sk_SimpleImputer) -> None:
        imputer.strategy = "mean"


class _Median(Imputer.Strategy):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Median):
            return NotImplemented
        return True

    def __hash__(self):
        return _structural_hash(str(self))

    def __str__(self) -> str:
        return "Median"

    def _apply(self, imputer: sk_SimpleImputer) -> None:
        imputer.strategy = "median"


class _Mode(Imputer.Strategy):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Mode):
            return NotImplemented
        return True

    def __hash__(self):
        return _structural_hash(str(self))

    def __str__(self) -> str:
        return "Mode"

    def _apply(self, imputer: sk_SimpleImputer) -> None:
        imputer.strategy = "most_frequent"


# Override the methods with classes, so they can be used in `isinstance` calls. Unlike methods, classes define a type.
# This is needed for the DSL, where imputer strategies are modeled using an enum.
Imputer.Strategy.Constant = _Constant
Imputer.Strategy.Mean = _Mean
Imputer.Strategy.Median = _Median
Imputer.Strategy.Mode = _Mode

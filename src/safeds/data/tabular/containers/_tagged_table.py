from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from safeds.data.tabular.containers import Column, Row, Table
from safeds.exceptions import (
    ColumnIsTargetError,
    IllegalSchemaModificationError,
    UnknownColumnNameError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any


class TaggedTable(Table):
    """
    A tagged table is a table that additionally knows which columns are features and which are the target to predict.

    Parameters
    ----------
    data : Mapping[str, Sequence[Any]]
        The data.
    target_name : str
        Name of the target column.
    feature_names : list[str] | None
        Names of the feature columns. If None, all columns except the target column are used.

    Raises
    ------
    ColumnLengthMismatchError
        If columns have different lengths.
    ValueError
        If the target column is also a feature column.
    ValueError
        If no feature columns are specified.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Table, TaggedTable
    >>> table = Table({"col1": ["a", "b"], "col2": [1, 2]})
    >>> tagged_table = table.tag_columns("col2", ["col1"])
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_table(
        table: Table,
        target_name: str,
        feature_names: list[str] | None = None,
    ) -> TaggedTable:
        """
        Create a tagged table from a table.

        Parameters
        ----------
        table : Table
            The table.
        target_name : str
            Name of the target column.
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target column are used.

        Returns
        -------
        tagged_table : TaggedTable
            The created table.

        Raises
        ------
        UnknownColumnNameError
            If target_name matches none of the column names.
        ValueError
            If the target column is also a feature column.
        ValueError
            If no feature columns are specified.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table, TaggedTable
        >>> table = Table({"col1": ["a", "b", "c", "a"], "col2": [1, 2, 3, 4]})
        >>> tagged_table = TaggedTable._from_table(table, "col2", ["col1"])
        """
        table = table._as_table()
        if target_name not in table.column_names:
            raise UnknownColumnNameError([target_name])

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = table.column_names
            feature_names.remove(target_name)

        # Validate inputs
        if target_name in feature_names:
            raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        # Create result
        result = object.__new__(TaggedTable)

        result._data = table._data
        result._schema = table.schema
        result._features = table.keep_only_columns(feature_names)
        result._target = table.get_column(target_name)

        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        data: Mapping[str, Sequence[Any]],
        target_name: str,
        feature_names: list[str] | None = None,
    ):
        """
        Create a tagged table from a mapping of column names to their values.

        Parameters
        ----------
        data : Mapping[str, Sequence[Any]]
            The data.
        target_name : str
            Name of the target column.
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target column are used.

        Raises
        ------
        ColumnLengthMismatchError
            If columns have different lengths.
        ValueError
            If the target column is also a feature column.
        ValueError
            If no feature columns are specified.

        Examples
        --------
        >>> from safeds.data.tabular.containers import TaggedTable
        >>> table = TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"])
        """
        super().__init__(data)
        _data = Table(data)

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = _data.column_names
            if target_name in feature_names:
                feature_names.remove(target_name)

        # Validate inputs
        if target_name in feature_names:
            raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        self._features: Table = _data.keep_only_columns(feature_names)
        self._target: Column = _data.get_column(target_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def features(self) -> Table:
        """
        Get the feature columns of the tagged table.

        Returns
        -------
        Table
            The table containing the feature columns.
        """
        return self._features

    @property
    def target(self) -> Column:
        """
        Get the target column of the tagged table.

        Returns
        -------
        Column
            The target column.
        """
        return self._target

    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _copy(self) -> TaggedTable:
        """
        Return a copy of this tagged table.

        Returns
        -------
        table : TaggedTable
            The copy of this tagged table.
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------------------------------------------------------------
    # Specific methods from TaggedTable class:
    # ------------------------------------------------------------------------------------------------------------------

    def add_column_as_feature(self, column: Column) -> TaggedTable:
        """
        Return a new table with the provided column attached at the end, as a feature column.

        the original table is not modified.

        Parameters
        ----------
        column : Column
            The column to be added.

        Returns
        -------
        result : TaggedTable
            The table with the attached feature column.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.
        """
        return TaggedTable._from_table(
            super().add_column(column),
            target_name=self.target.name,
            feature_names=[*self.features.column_names, column.name],
        )

    def add_columns_as_features(self, columns: list[Column] | Table) -> TaggedTable:
        """
        Return a new `TaggedTable` with the provided columns attached at the end, as feature columns.

        The original table is not modified.

        Parameters
        ----------
        columns : list[Column] | Table
            The columns to be added as features.

        Returns
        -------
        result : TaggedTable
            The table with the attached feature columns.

        Raises
        ------
        DuplicateColumnNameError
            If any of the new feature columns already exist.
        ColumnSizeError
            If the size of any feature column does not match the number of rows.
        """
        return TaggedTable._from_table(
            super().add_columns(columns),
            target_name=self.target.name,
            feature_names=self.features.column_names
            + [col.name for col in (columns.to_columns() if isinstance(columns, Table) else columns)],
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Overriden methods from Table class:
    # ------------------------------------------------------------------------------------------------------------------

    def _as_table(self: TaggedTable) -> Table:
        """
        Return a new `Table` with the tagging removed.

        The original TaggedTable is not modified.

        Parameters
        ----------
        self: TaggedTable
            The TaggedTable.

        Returns
        -------
        table: Table
            The table as an untagged Table, i.e. without the information about which columns are features or target.

        """
        return Table.from_columns(super().to_columns())

    def add_column(self, column: Column) -> TaggedTable:
        """
        Return a new `TaggedTable` with the provided column attached at the end, as neither target nor feature column.

        The original table is not modified.

        Parameters
        ----------
        column : Column
            The column to be added.

        Returns
        -------
        result : TaggedTable
            The table with the column attached as neither target nor feature column.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.
        """
        return TaggedTable._from_table(
            super().add_column(column),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def add_columns(self, columns: list[Column] | Table) -> TaggedTable:
        """
        Return a new `TaggedTable` with multiple added columns, as neither target nor feature columns.

        The original table is not modified.

        Parameters
        ----------
        columns : list[Column] or Table
            The columns to be added.

        Returns
        -------
        result: TaggedTable
            A new table combining the original table and the given columns as neither target nor feature columns.

        Raises
        ------
        DuplicateColumnNameError
            If at least one column name from the provided column list already exists in the table.
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the table.
        """
        return TaggedTable._from_table(
            super().add_columns(columns),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def add_row(self, row: Row) -> TaggedTable:
        """
        Return a new `TaggedTable` with an added Row attached.

        The original table is not modified.

        Parameters
        ----------
        row : Row
            The row to be added.

        Returns
        -------
        table : TaggedTable
            A new table with the added row at the end.

        Raises
        ------
        UnknownColumnNameError
            If the row has different column names than the table.
        """
        return TaggedTable._from_table(super().add_row(row), target_name=self.target.name)

    def add_rows(self, rows: list[Row] | Table) -> TaggedTable:
        """
        Return a new `TaggedTable` with multiple added Rows attached.

        The original table is not modified.

        Parameters
        ----------
        rows : list[Row] or Table
            The rows to be added.

        Returns
        -------
        result : TaggedTable
            A new table which combines the original table and the given rows.

        Raises
        ------
        UnknownColumnNameError
            If at least one of the rows have different column names than the table.
        """
        return TaggedTable._from_table(super().add_rows(rows), target_name=self.target.name)

    def filter_rows(self, query: Callable[[Row], bool]) -> TaggedTable:
        """
        Return a new `TaggedTable` containing only rows that match the given Callable (e.g. lambda function).

        The original table is not modified.

        Parameters
        ----------
        query : lambda function
            A Callable that is applied to all rows.

        Returns
        -------
        table : TaggedTable
            A table containing only the rows to match the query.
        """
        return TaggedTable._from_table(
            super().filter_rows(query),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def keep_only_columns(self, column_names: list[str]) -> TaggedTable:
        """
        Return a new `TaggedTable` with only the given column(s).

        The original table is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing only the columns to be kept.

        Returns
        -------
        table : TaggedTable
            A table containing only the given column(s).

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        IllegalSchemaModificationError
            If none of the given columns is the target column or any of the feature columns.
        """
        if self.target.name not in column_names:
            raise IllegalSchemaModificationError("Must keep the target column.")
        if len(set(self.features.column_names).intersection(set(column_names))) == 0:
            raise IllegalSchemaModificationError("Must keep at least one feature column.")
        return TaggedTable._from_table(
            super().keep_only_columns(column_names),
            target_name=self.target.name,
            feature_names=sorted(
                set(self.features.column_names).intersection(set(column_names)),
                key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
            ),
        )

    def remove_columns(self, column_names: list[str]) -> TaggedTable:
        """
        Return a new `TaggedTable` with the given column(s) removed from the table.

        The original table is not modified.

        Parameters
        ----------
        column_names : list[str]
            The names of all columns to be dropped.

        Returns
        -------
        table : TaggedTable
            A table without the given columns.

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        ColumnIsTargetError
            If any of the given columns is the target column.
        IllegalSchemaModificationError
            If the given columns contain all the feature columns.
        """
        if self.target.name in column_names:
            raise ColumnIsTargetError(self.target.name)
        if len(set(self.features.column_names) - set(column_names)) == 0:
            raise IllegalSchemaModificationError("You cannot remove every feature column.")
        return TaggedTable._from_table(
            super().remove_columns(column_names),
            target_name=self.target.name,
            feature_names=sorted(
                set(self.features.column_names) - set(column_names),
                key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
            ),
        )

    def remove_columns_with_missing_values(self) -> TaggedTable:
        """
        Return a new `TaggedTable` with every column that misses values removed.

        The original table is not modified.

        Returns
        -------
        table : TaggedTable
            A table without the columns that contain missing values.

        Raises
        ------
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        IllegalSchemaModificationError
            If the columns to remove contain all the feature columns.
        """
        table = super().remove_columns_with_missing_values()
        if self.target.name not in table.column_names:
            raise ColumnIsTargetError(self.target.name)
        if len(set(self.features.column_names).intersection(set(table.column_names))) == 0:
            raise IllegalSchemaModificationError("You cannot remove every feature column.")
        return TaggedTable._from_table(
            table,
            self.target.name,
            feature_names=sorted(
                set(self.features.column_names).intersection(set(table.column_names)),
                key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
            ),
        )

    def remove_columns_with_non_numerical_values(self) -> TaggedTable:
        """
        Return a new `TaggedTable` with every column that contains non-numerical values removed.

        The original table is not modified.

        Returns
        -------
        table : TaggedTable
            A table without the columns that contain non-numerical values.

        Raises
        ------
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        IllegalSchemaModificationError
            If the columns to remove contain all the feature columns.
        """
        table = super().remove_columns_with_non_numerical_values()
        if self.target.name not in table.column_names:
            raise ColumnIsTargetError(self.target.name)
        if len(set(self.features.column_names).intersection(set(table.column_names))) == 0:
            raise IllegalSchemaModificationError("You cannot remove every feature column.")
        return TaggedTable._from_table(
            table,
            self.target.name,
            feature_names=sorted(
                set(self.features.column_names).intersection(set(table.column_names)),
                key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
            ),
        )

    def remove_duplicate_rows(self) -> TaggedTable:
        """
        Return a new `TaggedTable` with all row duplicates removed.

        The original table is not modified.

        Returns
        -------
        result : TaggedTable
            The table with the duplicate rows removed.
        """
        return TaggedTable._from_table(
            super().remove_duplicate_rows(),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def remove_rows_with_missing_values(self) -> TaggedTable:
        """
        Return a new `TaggedTable` without the rows that contain missing values.

        The original table is not modified.

        Returns
        -------
        table : TaggedTable
            A table without the rows that contain missing values.
        """
        return TaggedTable._from_table(
            super().remove_rows_with_missing_values(),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def remove_rows_with_outliers(self) -> TaggedTable:
        """
        Return a new `TaggedTable` with all rows that contain at least one outlier removed.

        We define an outlier as a value that has a distance of more than 3 standard deviations from the column mean.
        Missing values are not considered outliers. They are also ignored during the calculation of the standard
        deviation.

        The original table is not modified.

        Returns
        -------
        new_table : TaggedTable
            A new table without rows containing outliers.
        """
        return TaggedTable._from_table(
            super().remove_rows_with_outliers(),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def rename_column(self, old_name: str, new_name: str) -> TaggedTable:
        """
        Return a new `TaggedTable` with a single column renamed.

        The original table is not modified.

        Parameters
        ----------
        old_name : str
            The old name of the target column.
        new_name : str
            The new name of the target column.

        Returns
        -------
        table : TaggedTable
            The Table with the renamed column.

        Raises
        ------
        UnknownColumnNameError
            If the specified old target column name does not exist.
        DuplicateColumnNameError
            If the specified new target column name already exists.
        """
        return TaggedTable._from_table(
            super().rename_column(old_name, new_name),
            target_name=new_name if self.target.name == old_name else self.target.name,
            feature_names=(
                self.features.column_names
                if old_name not in self.features.column_names
                else [
                    column_name if column_name != old_name else new_name for column_name in self.features.column_names
                ]
            ),
        )

    def replace_column(self, old_column_name: str, new_columns: list[Column]) -> TaggedTable:
        """
        Return a new `TaggedTable` with the specified old column replaced by a list of new columns.

        If the column to be replaced is the target column, it must be replaced by exactly one column. That column
        becomes the new target column. If the column to be replaced is a feature column, the new columns that replace it
        all become feature columns.

        The order of columns is kept. The original table is not modified.

        Parameters
        ----------
        old_column_name : str
            The name of the column to be replaced.
        new_columns : list[Column]
            The new columns replacing the old column.

        Returns
        -------
        result : TaggedTable
            A table with the old column replaced by the new column.

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist.
        DuplicateColumnNameError
            If the new column already exists and the existing column is not affected by the replacement.
        ColumnSizeError
            If the size of the column does not match the amount of rows.
        IllegalSchemaModificationError
            If the target column would be removed or replaced by more than one column.
        """
        if old_column_name == self.target.name:
            if len(new_columns) != 1:
                raise IllegalSchemaModificationError(
                    f'Target column "{self.target.name}" can only be replaced by exactly one new column.',
                )
            else:
                return TaggedTable._from_table(
                    super().replace_column(old_column_name, new_columns),
                    target_name=new_columns[0].name,
                    feature_names=self.features.column_names,
                )
        else:
            return TaggedTable._from_table(
                super().replace_column(old_column_name, new_columns),
                target_name=self.target.name,
                feature_names=(
                    self.features.column_names
                    if old_column_name not in self.features.column_names
                    else self.features.column_names[: self.features.column_names.index(old_column_name)]
                    + [col.name for col in new_columns]
                    + self.features.column_names[self.features.column_names.index(old_column_name) + 1 :]
                ),
            )

    def shuffle_rows(self) -> TaggedTable:
        """
        Return a new `TaggedTable` with randomly shuffled rows of this table.

        The original table is not modified.

        Returns
        -------
        result : TaggedTable
            The shuffled Table.
        """
        return TaggedTable._from_table(
            super().shuffle_rows(),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def slice_rows(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
    ) -> TaggedTable:
        """
        Slice a part of the table into a new `TaggedTable`.

        The original table is not modified.

        Parameters
        ----------
        start : int | None
            The first index of the range to be copied into a new table, None by default.
        end : int | None
            The last index of the range to be copied into a new table, None by default.
        step : int
            The step size used to iterate through the table, 1 by default.

        Returns
        -------
        result : TaggedTable
            The resulting table.

        Raises
        ------
        IndexOutOfBoundsError
            If the index is out of bounds.
        """
        return TaggedTable._from_table(
            super().slice_rows(start, end, step),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def sort_columns(
        self,
        comparator: Callable[[Column, Column], int] = lambda col1, col2: (col1.name > col2.name)
        - (col1.name < col2.name),
    ) -> TaggedTable:
        """
        Sort the columns of a `TaggedTable` with the given comparator and return a new `TaggedTable`.

        The comparator is a function that takes two columns `col1` and `col2` and
        returns an integer:

        * If the function returns a negative number, `col1` will be ordered before `col2`.
        * If the function returns a positive number, `col1` will be ordered after `col2`.
        * If the function returns 0, the original order of `col1` and `col2` will be kept.

        If no comparator is given, the columns will be sorted alphabetically by their name.

        The original table is not modified.

        Parameters
        ----------
        comparator : Callable[[Column, Column], int]
            The function used to compare two columns.

        Returns
        -------
        new_table : TaggedTable
            A new table with sorted columns.
        """
        sorted_table = super().sort_columns(comparator)
        return TaggedTable._from_table(
            sorted_table,
            target_name=self.target.name,
            feature_names=sorted(
                set(sorted_table.column_names).intersection(self.features.column_names),
                key={val: ix for ix, val in enumerate(sorted_table.column_names)}.__getitem__,
            ),
        )

    def sort_rows(self, comparator: Callable[[Row, Row], int]) -> TaggedTable:
        """
        Sort the rows of a `TaggedTable` with the given comparator and return a new `TaggedTable`.

        The comparator is a function that takes two rows `row1` and `row2` and
        returns an integer:

        * If the function returns a negative number, `row1` will be ordered before `row2`.
        * If the function returns a positive number, `row1` will be ordered after `row2`.
        * If the function returns 0, the original order of `row1` and `row2` will be kept.

        The original table is not modified.

        Parameters
        ----------
        comparator : Callable[[Row, Row], int]
            The function used to compare two rows.

        Returns
        -------
        new_table : TaggedTable
            A new table with sorted rows.
        """
        return TaggedTable._from_table(
            super().sort_rows(comparator),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def transform_column(self, name: str, transformer: Callable[[Row], Any]) -> TaggedTable:
        """
        Return a new `TaggedTable` with the provided column transformed by calling the provided transformer.

        The original table is not modified.

        Returns
        -------
        result : TaggedTable
            The table with the transformed column.

        Raises
        ------
        UnknownColumnNameError
            If the column does not exist.
        """
        return TaggedTable._from_table(
            super().transform_column(name, transformer),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

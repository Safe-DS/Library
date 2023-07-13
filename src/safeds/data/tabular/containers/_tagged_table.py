from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pandas as pd

from safeds.data.tabular.containers import Column, Row, Table
from safeds.exceptions import (
    ColumnIsTargetError,
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IllegalSchemaModificationError,
    IndexOutOfBoundsError,
    SchemaMismatchError,
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

        # Validate column lengths
        expected_length: int | None = None
        for column_values in data.values():
            if expected_length is None:
                expected_length = len(column_values)
            elif len(column_values) != expected_length:
                raise ColumnLengthMismatchError(
                    "\n".join(
                        f"{column_name}: {len(column_values)}" for column_name, column_values in data.items())
                )

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
        Return the original table with the provided column attached at the end, as a feature column.

        This table is not modified.

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
        if column.name in self.column_names:
            raise DuplicateColumnNameError(f"Column '{column.name}' already exists.")

        if column.number_of_rows != self.number_of_rows and self.number_of_columns != 0:
            raise ColumnSizeError(str(self.number_of_rows), str(column.number_of_rows))

        return TaggedTable._from_table(
            super().add_column(column),
            target_name=self.target.name,
            feature_names=[*self.features.column_names, column.name],
        )

    def add_columns_as_features(self, columns: list[Column] | Table) -> TaggedTable:
        """
        Return the original table with the provided columns attached at the end, as feature columns.

        This table is not modified.

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
        if isinstance(columns, Table):
            columns = columns.to_columns()

        for column in columns:
            if column.name in self.column_names:
                raise DuplicateColumnNameError(column.name)

            if column.number_of_rows != self.number_of_rows and self.number_of_columns != 0:
                raise ColumnSizeError(str(self.number_of_rows), str(column.number_of_rows))

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
        Remove the tagging from a TaggedTable.

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
        Return the original table with the provided column attached at the end, as neither target nor feature column.

        This table is not modified.

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
            If the size of the column does not match the amount of rows.
        """
        if column.name in self.column_names:
            raise DuplicateColumnNameError(f"Column '{column.name}' already exists.")

        if column.number_of_rows != self.number_of_rows and self.number_of_columns != 0:
            raise ColumnSizeError(str(self.number_of_rows), str(column.number_of_rows))

        return TaggedTable._from_table(
            super().add_column(column),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def add_columns(self, columns: list[Column] | Table) -> TaggedTable:
        """
        Add multiple columns to the table, as neither target nor feature columns.

        This table is not modified.

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
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the table.
        DuplicateColumnNameError
            If at least one column name from the provided column list already exists in the table.
        """
        if isinstance(columns, Table):
            columns = columns.to_columns()

        for column in columns:
            if column.name in self.column_names:
                raise DuplicateColumnNameError(f"Column '{column.name}' already exists.")

            if column.number_of_rows != self.number_of_rows and self.number_of_columns != 0:
                raise ColumnSizeError(str(self.number_of_rows), str(column.number_of_rows))

        return TaggedTable._from_table(
            super().add_columns(columns),
            target_name=self.target.name,
            feature_names=self.features.column_names,
        )

    def add_row(self, row: Row) -> TaggedTable:
        """
        Add a row to the table.

        This table is not modified.

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
        SchemaMismatchError
            If the schema of the row does not match the table schema.
        """
        if self.number_of_rows == 0:
            if self.number_of_columns == 0:
                for column in row.column_names:
                    self._data[column] = Column(column, [])
                self._schema = Table._from_pandas_dataframe(pd.DataFrame(columns=row.column_names))._schema
            elif self.column_names != row.column_names:
                raise SchemaMismatchError
        elif self._schema != row.schema:
            raise SchemaMismatchError

        return TaggedTable._from_table(super().add_row(row), target_name=self.target.name)

    def add_rows(self, rows: list[Row] | Table) -> TaggedTable:
        """
        Add multiple rows to the table.

        This table is not modified.

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
        SchemaMismatchError
            If the schema of on of the row does not match the table schema.
        """
        if isinstance(rows, Table):
            rows = rows.to_rows()
        for row in rows:
            if self.number_of_rows == 0:
                if self.number_of_columns == 0:
                    for column in row.column_names:
                        self._data[column] = Column(column, [])
                    self._schema = Table._from_pandas_dataframe(pd.DataFrame(columns=self.column_names))._schema
                elif self.column_names != row.column_names:
                    raise SchemaMismatchError
            elif self._schema != row.schema:
                raise SchemaMismatchError

        return TaggedTable._from_table(super().add_rows(rows), target_name=self.target.name)

    def filter_rows(self, query: Callable[[Row], bool]) -> TaggedTable:
        """
        Return a table containing only rows that match the given Callable (e.g. lambda function).

        This table is not modified.

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
        Return a table with only the given column(s).

        This table is not modified.

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
        invalid_columns = []
        for name in column_names:
            if not self._schema.has_column(name):
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns)

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
        Remove the given column(s) from the table.

        This table is not modified.

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
        invalid_columns = []
        for name in column_names:
            if not self._schema.has_column(name):
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns)

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
        Remove every column that misses values.

        This table is not modified.

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
        Remove every column that contains non-numerical values.

        This table is not modified.

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
        Remove all row duplicates.

        This table is not modified.

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
        Return a table without the rows that contain missing values.

        This table is not modified.

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
        Remove all rows from the table that contain at least one outlier.

        We define an outlier as a value that has a distance of more than 3 standard deviations from the column mean.
        Missing values are not considered outliers. They are also ignored during the calculation of the standard
        deviation.

        This table is not modified.

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
        Rename a single column.

        This table is not modified.

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
        if old_name not in self._schema.column_names:
            raise UnknownColumnNameError([old_name])
        if old_name == new_name:
            return self
        if new_name in self._schema.column_names:
            raise DuplicateColumnNameError(new_name)

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
        Replace the specified old column by a list of new columns.

        If the column to be replaced is the target column, it must be replaced by exactly one column. That column
        becomes the new target column. If the column to be replaced is a feature column, the new columns that replace it
        all become feature columns.

        The order of columns is kept. This table is not modified.

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
        if old_column_name not in self._schema.column_names:
            raise UnknownColumnNameError([old_column_name])

        columns = list[Column]()
        for old_column in self.column_names:
            if old_column == old_column_name:
                for new_column in new_columns:
                    if new_column.name in self.column_names and new_column.name != old_column_name:
                        raise DuplicateColumnNameError(new_column.name)

                    if self.number_of_rows != new_column.number_of_rows:
                        raise ColumnSizeError(str(self.number_of_rows), str(new_column.number_of_rows))
                    columns.append(new_column)
            else:
                columns.append(self.get_column(old_column))

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
                    + self.features.column_names[self.features.column_names.index(old_column_name) + 1:]
                ),
            )

    def shuffle_rows(self) -> TaggedTable:
        """
        Shuffle the table randomly.

        This table is not modified.

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
        Slice a part of the table into a new table.

        This table is not modified.

        Parameters
        ----------
        start : int
            The first index of the range to be copied into a new table, None by default.
        end : int
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
        if start is None:
            start = 0

        if end is None:
            end = self.number_of_rows

        if end < start:
            raise IndexOutOfBoundsError(slice(start, end))
        if start < 0 or end < 0 or start > self.number_of_rows or end > self.number_of_rows:
            raise IndexOutOfBoundsError(start if start < 0 or start > self.number_of_rows else end)

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

        This table is not modified.

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

        This table is not modified.

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
        Transform provided column by calling provided transformer.

        This table is not modified.

        Returns
        -------
        result : TaggedTable
            The table with the transformed column.

        Raises
        ------
        UnknownColumnNameError
            If the column does not exist.
        """
        if self.has_column(name):
            items: list = [transformer(item) for item in self.to_rows()]
            result: list[Column] = [Column(name, items)]
            return TaggedTable._from_table(
                super().replace_column(name, result),
                target_name=self.target.name,
                feature_names=self.features.column_names,
            )
        raise UnknownColumnNameError([name])

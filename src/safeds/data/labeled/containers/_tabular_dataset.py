from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    UnknownColumnNameError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset


class TabularDataset:
    """
    A tabular dataset maps feature columns to a target column.

    Parameters
    ----------
    data:
        The data.
    target_name:
        Name of the target column.
    feature_names:
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
    >>> from safeds.data.tabular.containers import Table
    >>> table = Table({"col1": ["a", "b"], "col2": [1, 2]})
    >>> tabular_dataset = table.to_tabular_dataset("col2", ["col1"])
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_table(
        table: Table,
        target_name: str,
        feature_names: list[str] | None = None,
    ) -> TabularDataset:
        """
        Create a tabular dataset from a table.

        Parameters
        ----------
        table:
            The table.
        target_name:
            Name of the target column.
        feature_names:
            Names of the feature columns. If None, all columns except the target column are used.

        Returns
        -------
        tabular_dataset:
            The created tabular dataset.

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
        >>> from safeds.data.labeled.containers import TabularDataset
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"col1": ["a", "b", "c", "a"], "col2": [1, 2, 3, 4]})
        >>> tabular_dataset = TabularDataset._from_table(table, "col2", ["col1"])
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
        result = object.__new__(TabularDataset)

        result._table = table
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
        Create a tabular dataset from a mapping of column names to their values.

        Parameters
        ----------
        data:
            The data.
        target_name:
            Name of the target column.
        feature_names:
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
        >>> from safeds.data.labeled.containers import TabularDataset
        >>> table = TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"])
        """
        self._table = Table(data)

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = self._table.column_names
            if target_name in feature_names:
                feature_names.remove(target_name)

        # Validate inputs
        if target_name in feature_names:
            raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        self._features: Table = self._table.keep_only_columns(feature_names)
        self._target: Column = self._table.get_column(target_name)
        self._extras: Table = self._table.remove_columns([*feature_names, target_name])

    def __eq__(self, other: object) -> bool:
        """
        Compare two tabular datasets.

        Returns
        -------
        equals:
            'True' if features and targets are equal, 'False' otherwise.
        """
        if not isinstance(other, TabularDataset):
            return NotImplemented
        if self is other:
            return True
        return self.target == other.target and self.features == other.features and self._table == other._table

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this tabular dataset.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.target, self.features, self._table)

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._target) + sys.getsizeof(self._features) + sys.getsizeof(self._table)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def features(self) -> Table:
        """The feature columns of the tabular dataset."""
        return self._features

    @property
    def target(self) -> Column:
        """The target column of the tabular dataset."""
        return self._target

    @property
    def extras(self) -> Table:
        """
        Additional columns of the tabular dataset that are neither features nor target.

        These can be used to store additional information about instances, such as IDs.
        """
        return self._extras

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_table(self) -> Table:
        """
        Return a new `Table` containing the feature columns and the target column.

        The original `TabularDataset` is not modified.

        Returns
        -------
        table:
            A table containing the feature columns and the target column.
        """
        return self._table

    def _into_dataloader_with_classes(self, batch_size: int, num_of_classes: int) -> DataLoader:
        """
        Return a Dataloader for the data stored in this table, used for training neural networks.

        The original table is not modified.

        Parameters
        ----------
        batch_size:
            The size of data batches that should be loaded at one time.

        Returns
        -------
        result:
            The DataLoader.

        """
        import torch
        from torch.utils.data import DataLoader

        if num_of_classes <= 2:
            return DataLoader(
                dataset=_create_dataset(
                    torch.Tensor(self.features._data.values),
                    torch.Tensor(self.target._data).unsqueeze(dim=-1),
                ),
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            return DataLoader(
                dataset=_create_dataset(
                    torch.Tensor(self.features._data.values),
                    torch.nn.functional.one_hot(torch.LongTensor(self.target._data), num_classes=num_of_classes),
                ),
                batch_size=batch_size,
                shuffle=True,
            )

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the tabular dataset.

        Returns
        -------
        output:
            The generated HTML.
        """
        return self._table._repr_html_()


def _create_dataset(features: Tensor, target: Tensor) -> Dataset:
    import torch
    from torch.utils.data import Dataset

    class _CustomDataset(Dataset):
        def __init__(self, features: Tensor, target: Tensor):
            self.X = features.to(torch.float32)
            self.Y = target.to(torch.float32)
            self.len = self.X.size(dim=0)

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.X[item], self.Y[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features, target)

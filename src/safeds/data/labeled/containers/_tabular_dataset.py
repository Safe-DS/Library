from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._config import _get_device, _init_default_device
from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Column, Table

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset


class TabularDataset:
    """
    A tabular dataset maps feature columns to a target column.

    Create a tabular dataset from a mapping of column names to their values.

    Parameters
    ----------
    data:
        The data.
    target_name:
        Name of the target column.
    extra_names:
        Names of the columns that are neither features nor target. If None, no extra columns are used, i.e. all but
        the target column are used as features.

    Raises
    ------
    ColumnLengthMismatchError
        If columns have different lengths.
    ValueError
        If the target column is also an extra column.
    ValueError
        If no feature columns remains.

    Examples
    --------
    >>> from safeds.data.labeled.containers import TabularDataset
    >>> dataset = TabularDataset(
    ...     {"id": [1, 2, 3], "feature": [4, 5, 6], "target": [1, 2, 3]},
    ...     target_name="target",
    ...     extra_names=["id"]
    ... )
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        data: Table | Mapping[str, Sequence[Any]],
        target_name: str,
        extra_names: list[str] | None = None,
    ):
        # Preprocess inputs
        if not isinstance(data, Table):
            data = Table(data)
        if extra_names is None:
            extra_names = []

        # Derive feature names
        feature_names = [name for name in data.column_names if name not in {target_name, *extra_names}]

        # Validate inputs
        if target_name in extra_names:
            raise ValueError(f"Column '{target_name}' cannot be both target and extra.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must remain.")

        # Set attributes
        self._table: Table = data
        self._features: Table = data.keep_only_columns(feature_names)
        self._target: Column = data.get_column(target_name)
        self._extras: Table = data.keep_only_columns(extra_names)

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
        return self.target == other.target and self.features == other.features and self._extras == other._extras

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this tabular dataset.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.target, self.features, self._extras)

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._target) + sys.getsizeof(self._features) + sys.getsizeof(self._extras)

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

        _init_default_device()

        if num_of_classes <= 2:
            return DataLoader(
                dataset=_create_dataset(
                    torch.Tensor(self.features._data.values).to(_get_device()),
                    torch.Tensor(self.target._data).to(_get_device()).unsqueeze(dim=-1),
                ),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator(device=_get_device()),
            )
        else:
            return DataLoader(
                dataset=_create_dataset(
                    torch.Tensor(self.features._data.values).to(_get_device()),
                    torch.nn.functional.one_hot(
                        torch.LongTensor(self.target._data).to(_get_device()),
                        num_classes=num_of_classes,
                    ),
                ),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator(device=_get_device()),
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

    _init_default_device()

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

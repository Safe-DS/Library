from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from safeds._config import _get_device, _init_default_device
from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Column, Table

from ._dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset


class TabularDataset(Dataset[Table, Column]):
    """
    A dataset containing tabular data. It can be used to train machine learning models.

    Columns in a tabular dataset are divided into three categories:

    - The target column is the column that a model should predict.
    - Feature columns are columns that a model should use to make predictions.
    - Extra columns are columns that are neither feature nor target. They can be used to provide additional context,
      like an ID column.

    Feature columns are implicitly defined as all columns except the target and extra columns. If no extra columns
    are specified, all columns except the target column are used as features.

    Parameters
    ----------
    data:
        The data.
    target_name:
        The name of the target column.
    extra_names:
        Names of the columns that are neither features nor target. If None, no extra columns are used, i.e. all but
        the target column are used as features.

    Raises
    ------
    ColumnNotFoundError
        If a column name is not found in the data.
    ValueError
        If the target column is also an extra column.
    ValueError
        If no feature columns remains.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Table
    >>> table = Table(
    ...     {
    ...         "id": [1, 2, 3],
    ...         "feature": [4, 5, 6],
    ...         "target": [1, 2, 3],
    ...     },
    ... )
    >>> dataset = table.to_tabular_dataset(target_name="target", extra_names=["id"])
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        data: Table | Mapping[str, Sequence[Any]],
        target_name: str,
        *,
        extra_names: list[str] | None = None,
    ):
        from safeds.data.tabular.containers import Table

        # Preprocess inputs
        if not isinstance(data, Table):
            data = Table(data)
        if extra_names is None:
            extra_names = []

        # Derive feature names (build the set once, since comprehensions evaluate their condition every iteration)
        non_feature_names = {target_name, *extra_names}
        feature_names = [name for name in data.column_names if name not in non_feature_names]

        # Validate inputs
        if target_name in extra_names:
            raise ValueError(f"Column '{target_name}' cannot be both target and extra.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must remain.")

        # Set attributes
        self._table: Table = data
        self._features: Table = data.remove_columns_except(feature_names)
        self._target: Column = data.get_column(target_name)
        self._extras: Table = data.remove_columns_except(extra_names)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TabularDataset):
            return NotImplemented
        if self is other:
            return True
        return self.target == other.target and self.features == other.features and self._extras == other._extras

    def __hash__(self) -> int:
        return _structural_hash(self.target, self.features, self._extras)

    def __repr__(self) -> str:
        return self._table.__repr__()

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._target) + sys.getsizeof(self._features) + sys.getsizeof(self._extras)

    def __str__(self) -> str:
        return self._table.__str__()

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
        Return a table containing all columns of the tabular dataset.

        Returns
        -------
        table:
            A table containing all columns of the tabular dataset.
        """
        return self._table

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return a compact HTML representation of the tabular dataset for IPython.

        Returns
        -------
        html:
            The generated HTML.
        """
        return self._table._repr_html_()

    # TODO
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
                    self.features._data_frame.to_torch().to(_get_device()),
                    self.target._series.to_torch().to(_get_device()).unsqueeze(dim=-1),
                ),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator(device=_get_device()),
            )
        else:
            return DataLoader(
                dataset=_create_dataset(
                    self.features._data_frame.to_torch().to(_get_device()),
                    torch.nn.functional.one_hot(
                        self.target._series.to_torch().to(_get_device()),
                        num_classes=num_of_classes,
                    ),
                ),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator(device=_get_device()),
            )


# TODO
def _create_dataset(features: Tensor, target: Tensor) -> TorchDataset:
    import torch
    from torch.utils.data import Dataset as TorchDataset

    _init_default_device()

    class _CustomDataset(TorchDataset):
        def __init__(self, features: Tensor, target: Tensor):
            self.X = features.to(torch.float32)
            self.Y = target.to(torch.float32)
            self.len = self.X.size(dim=0)

        def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.X[item], self.Y[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features, target)

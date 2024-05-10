from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from safeds.data.tabular.containers import ExperimentalColumn, ExperimentalTable


class ExperimentalTabularDataset:
    """
    A dataset containing tabular data. It can be used to train machine learning models.

    Columns in a tabular dataset are divided into three categories:

    * The target column is the column that a model should predict.
    * Feature columns are columns that a model should use to make predictions.
    * Extra columns are columns that are neither feature nor target. They can be used to provide additional context,
      like an ID column.

    Feature columns are implicitly defined as all columns except the target and extra columns. If no extra columns
    are specified, all columns except the target column are used as features.

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
    KeyError
        If a column name is not found in the data.
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
        data: ExperimentalTable,
        target_name: str,
        extra_names: list[str] | None = None,
    ):
        # Preprocess inputs
        if extra_names is None:
            extra_names = []

        # Derive feature names
        non_feature_names = {target_name, *extra_names}  # perf: Comprehensions evaluate their condition every iteration
        feature_names = [name for name in data.column_names if name not in non_feature_names]

        # Validate inputs
        if target_name in extra_names:
            raise ValueError(f"Column '{target_name}' cannot be both target and extra.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must remain.")

        # Set attributes
        self._table: ExperimentalTable = data
        self._features: ExperimentalTable = data.remove_columns_except(feature_names)
        self._target: ExperimentalColumn = data.get_column(target_name)
        self._extras: ExperimentalTable = data.remove_columns_except(extra_names)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentalTabularDataset):
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
    def features(self) -> ExperimentalTable:
        """The feature columns of the tabular dataset."""
        return self._features

    @property
    def target(self) -> ExperimentalColumn:
        """The target column of the tabular dataset."""
        return self._target

    @property
    def extras(self) -> ExperimentalTable:
        """
        Additional columns of the tabular dataset that are neither features nor target.

        These can be used to store additional information about instances, such as IDs.
        """
        return self._extras

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_table(self) -> ExperimentalTable:
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

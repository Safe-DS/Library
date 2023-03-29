from typing import Iterable, Optional

from IPython.core.display_functions import DisplayHandle
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.typing import TableSchema


class TaggedTable(Table):
    """
    A tagged table is a table that additionally knows which columns are features and which are the target to predict.

    Parameters
    ----------
    data : Iterable
        The data.
    target_name : str
        Name of the target column.
    feature_names : Optional[list[str]]
        Names of the feature columns. If None, all columns except the target column are used.
    schema : Optional[TableSchema]
        The schema of the table. If not specified, the schema will be inferred from the data.
    """

    def __init__(
        self,
        data: Iterable,
        target_name: str,
        feature_names: Optional[list[str]] = None,
        schema: Optional[TableSchema] = None,
    ):
        super().__init__(data, schema)

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = self.get_column_names()
            if target_name in feature_names:
                feature_names.remove(target_name)

        # Validate inputs
        if target_name in feature_names:
            raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        self._features: Table = self.keep_only_columns(feature_names)
        self._target: Column = self.get_column(target_name)

    @property
    def features(self) -> Table:
        return self._features

    @property
    def target(self) -> Column:
        return self._target

    def __repr__(self) -> str:
        tmp = self._features.add_column(self._target)
        header_info = "Target Column is '" + self._target.name + "'\n"
        return header_info + tmp.__repr__()

    def __str__(self) -> str:
        tmp = self._features.add_column(self._target)
        header_info = "Target Column is '" + self._target.name + "'\n"
        return header_info + tmp.__str__()

    def _ipython_display_(self) -> DisplayHandle:
        """
        Return a display object for the column to be used in Jupyter Notebooks.

        Returns
        -------
        output : DisplayHandle
            Output object.
        """
        tmp = self._features.add_column(self._target)
        header_info = "Target Column is '" + self._target.name + "'\n"
        print(header_info)
        return tmp._ipython_display_()

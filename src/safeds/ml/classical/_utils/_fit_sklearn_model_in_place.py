from __future__ import annotations

from typing import Any

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    DatasetMissesDataError,
    LearningError,
    MissingValuesColumnError,
    NonNumericColumnError,
    PlainTableError,
)


def _fit_sklearn_model_in_place(model: Any, tabular_dataset: TabularDataset) -> None:
    """
    Fit a scikit-learn model in-place on the given tabular dataset.

    Parameters
    ----------
    model:
        Classifier or Regression from scikit-learn.
    tabular_dataset:
        The tabular dataset containing the feature and target vectors.

    Raises
    ------
    LearningError
        If the tabular dataset contains invalid values or if the training failed.
    TypeError
        If a table is passed instead of a tabular dataset.
    NonNumericColumnError
        If the training data contains non-numerical values.
    MissingValuesColumnError
        If the training data contains missing values.
    DatasetMissesDataError
        If the training data contains no rows.
    """
    if not isinstance(tabular_dataset, TabularDataset) and isinstance(tabular_dataset, Table):
        raise PlainTableError

    if tabular_dataset._table.number_of_rows == 0:
        raise DatasetMissesDataError

    non_numerical_column_names = set(tabular_dataset.features.column_names) - set(
        tabular_dataset.features.remove_non_numeric_columns().column_names,
    )
    if len(non_numerical_column_names) != 0:
        raise NonNumericColumnError(
            str(non_numerical_column_names),
            "You can use the LabelEncoder or OneHotEncoder to transform your non-numerical data to numerical"
            " data.\nThe OneHotEncoder should be used if you work with nominal data. If your data contains too many"
            " different values\nor is ordinal, you should use the LabelEncoder.",
        )

    null_containing_column_names = set(tabular_dataset.features.column_names) - set(
        tabular_dataset.features.remove_columns_with_missing_values().column_names,
    )
    if len(null_containing_column_names) != 0:
        raise MissingValuesColumnError(
            str(null_containing_column_names),
            "You can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
            " remove the missing values entirely you can use the method `Table.remove_rows_with_missing_values`.",
        )

    try:
        model.fit(
            tabular_dataset.features._data_frame,
            tabular_dataset.target._series,
        )
    except ValueError as exception:
        raise LearningError(str(exception)) from exception

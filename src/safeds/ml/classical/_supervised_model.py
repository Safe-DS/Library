from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    LearningError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    PlainTableError,
    PredictionError,
)

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin, RegressorMixin

    from safeds.data.tabular.typing import DataType, Schema


class SupervisedModel(ABC):
    """A model for supervised learning tasks."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # The decorator is needed so the class really cannot be instantiated
    @abstractmethod
    def __init__(self) -> None:
        self._feature_schema: Schema | None = None
        self._target_name: str | None = None
        self._target_type: DataType | None = None
        self._wrapped_model: ClassifierMixin | RegressorMixin | None = None

    # The decorator ensures that the method is overridden in all subclasses
    @abstractmethod
    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._feature_schema,
            self._target_name,
            self._target_type,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return None not in (self._feature_schema, self._target_name, self._target_type, self._wrapped_model)

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and prediction
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, training_set: TabularDataset) -> Self:
        """
        Create a copy of this model and fit it with the given training data.

        **Note:** This model is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the features and target.

        Returns
        -------
        fitted_model:
            The fitted model.

        Raises
        ------
        PlainTableError
            If a table is passed instead of a TabularDataset.
        DatasetMissesDataError
            If the given training set contains no data.
        FittingWithChoiceError
            When trying to call this method on a model with hyperparameter choices.
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        if not isinstance(training_set, TabularDataset) and isinstance(training_set, Table):
            raise PlainTableError
        if training_set.to_table().row_count == 0:
            raise DatasetMissesDataError

        self._check_additional_fit_preconditions()
        self._check_more_additional_fit_preconditions(training_set)

        wrapped_model = self._get_sklearn_model()
        _fit_sklearn_model_in_place(wrapped_model, training_set)

        result = self._clone()
        result._feature_schema = training_set.features.schema
        result._target_name = training_set.target.name
        result._target_type = training_set.target.type
        result._wrapped_model = wrapped_model

        return result

    def predict(
        self,
        dataset: Table | TabularDataset,
    ) -> TabularDataset:
        """
        Predict the target values on the given dataset.

        **Note:** The model must be fitted.

        Parameters
        ----------
        dataset:
            The dataset containing at least the features.

        Returns
        -------
        prediction:
            The given dataset with an additional column for the predicted target values.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        DatasetMissesFeaturesError
            If the dataset misses feature columns.
        PredictionError
            If predicting with the given dataset failed.
        """
        self._check_additional_predict_preconditions(dataset)

        return _predict_with_sklearn_model(
            self._wrapped_model,
            dataset,
            self.get_feature_names(),
            self.get_target_name(),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """
        Return the names of the feature columns.

        **Note:** The model must be fitted.

        Returns
        -------
        feature_names:
            The names of the feature columns.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        # Used in favor of is_fitted, so the type checker is happy
        if self._feature_schema is None:
            raise ModelNotFittedError

        return self._feature_schema.column_names

    def get_features_schema(self) -> Schema:
        """
        Return the schema of the feature columns.

        **Note:** The model must be fitted.

        Returns
        -------
        feature_schema:
            The schema of the feature columns.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        # Used in favor of is_fitted, so the type checker is happy
        if self._feature_schema is None:
            raise ModelNotFittedError

        return self._feature_schema

    def get_target_name(self) -> str:
        """
        Return the name of the target column.

        **Note:** The model must be fitted.

        Returns
        -------
        target_name:
            The name of the target column.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        # Used in favor of is_fitted, so the type checker is happy
        if self._target_name is None:
            raise ModelNotFittedError

        return self._target_name

    def get_target_type(self) -> DataType:
        """
        Return the type of the target column.

        **Note:** The model must be fitted.

        Returns
        -------
        target_type:
            The type of the target column.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """
        # Used in favor of is_fitted, so the type checker is happy
        if self._target_type is None:
            raise ModelNotFittedError

        return self._target_type

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self) -> None:  # noqa: B027
        """Check additional preconditions for fitting the model and raise an error if any are violated."""

    def _check_more_additional_fit_preconditions(self, training_set: TabularDataset) -> None:  # noqa: B027
        """Check additional preconditions for fitting the model and raise an error if any are violated."""

    def _check_additional_fit_by_exhaustive_search_preconditions(self) -> None:  # noqa: B027
        """Check additional preconditions for fitting by exhaustive search and raise an error if any are violated."""

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset) -> None:  # noqa: B027
        """
        Check additional preconditions for predicting with the model and raise an error if any are violated.

        Parameters
        ----------
        dataset:
            The dataset containing at least the features.
        """

    def _get_models_for_all_choices(self) -> list[Self]:
        """Get a list of all possible models, given the Parameter Choices."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _clone(self) -> Self:
        """
        Return a new instance of this model with the same hyperparameters.

        Returns
        -------
        clone:
            A new instance of this model.
        """

    @abstractmethod
    def _get_sklearn_model(self) -> ClassifierMixin | RegressorMixin:
        """
        Return a new scikit-learn model that implements the algorithm of this model.

        Returns
        -------
        sklearn_model:
            The scikit-learn model.
        """


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


def _predict_with_sklearn_model(
    model: Any,
    dataset: Table | TabularDataset,
    feature_names: list[str] | None,
    target_name: str | None,
) -> TabularDataset:
    """
    Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

    Parameters
    ----------
    model:
        Classifier or regressor from scikit-learn.
    dataset:
        The dataset containing the features.
    target_name:
        The name of the target column.
    feature_names:
        The names of the feature columns.

    Returns
    -------
    table:
        A dataset containing the given features and the predicted target.

    Raises
    ------
    ModelNotFittedError
        If the model has not been fitted yet.
    DatasetMissesFeaturesError
        If the dataset misses feature columns.
    PredictionError
        If predicting with the given dataset failed.
    NonNumericColumnError
        If the dataset contains non-numerical values.
    MissingValuesColumnError
        If the dataset contains missing values.
    DatasetMissesDataError
        If the dataset contains no rows.
    """
    # Validation
    if model is None or target_name is None or feature_names is None:
        raise ModelNotFittedError
    if isinstance(dataset, TabularDataset):  # pragma: no cover
        dataset = dataset.features

    missing_feature_names = [feature_name for feature_name in feature_names if not dataset.has_column(feature_name)]
    if missing_feature_names:
        raise DatasetMissesFeaturesError(missing_feature_names)

    if dataset.row_count == 0:
        raise DatasetMissesDataError

    features = dataset.remove_columns_except(feature_names)

    non_numerical_column_names = set(features.column_names) - set(
        features.remove_non_numeric_columns().column_names,
    )
    if len(non_numerical_column_names) != 0:
        raise NonNumericColumnError(
            str(non_numerical_column_names),
            "You can use the LabelEncoder or OneHotEncoder to transform your non-numerical data to numerical"
            " data.\nThe OneHotEncoder should be used if you work with nominal data. If your data contains too many"
            " different values\nor is ordinal, you should use the LabelEncoder.\n",
        )

    null_containing_column_names = set(features.column_names) - set(
        features.remove_columns_with_missing_values().column_names,
    )
    if len(null_containing_column_names) != 0:
        raise MissingValuesColumnError(
            str(null_containing_column_names),
            "You can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
            " remove the missing values entirely you can use the method `Table.remove_rows_with_missing_values`.",
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            predicted_target_vector = model.predict(features._data_frame)
        output = dataset.remove_columns(target_name, ignore_unknown_names=True).add_columns(
            Column(target_name, predicted_target_vector),
        )

        extra_names = [
            column_name
            for column_name in dataset.column_names
            if column_name != target_name and column_name not in feature_names
        ]

        return TabularDataset(
            output,
            target_name=target_name,
            extra_names=extra_names,
        )
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception

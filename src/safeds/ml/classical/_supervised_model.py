from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import ModelNotFittedError
from safeds.ml.classical._utils import _fit_sklearn_model_in_place, _predict_with_sklearn_model

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin, RegressorMixin

    from safeds.data.tabular.containers import Table
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
    # Machine learning
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
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        self._check_additional_fit_preconditions(training_set)

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
        return self.get_features_schema().column_names

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

    def _check_additional_fit_preconditions(self, training_set: TabularDataset):  # noqa: B027
        """
        Check additional preconditions for fitting the model and raise an error if any are violated.

        Parameters
        ----------
        training_set:
            The training data containing the features and target.
        """

    def _check_additional_predict_preconditions(self, dataset: Table | TabularDataset):  # noqa: B027
        """
        Check additional preconditions for predicting with the model and raise an error if any are violated.

        Parameters
        ----------
        dataset:
            The dataset containing at least the features.
        """

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


def _extract_table(table_or_dataset: Table | TabularDataset) -> Table:
    """Extract the table from the given table or dataset."""
    if isinstance(table_or_dataset, TabularDataset):
        return table_or_dataset.to_table()
    else:
        return table_or_dataset

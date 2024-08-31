from __future__ import annotations

from abc import ABC
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait
from typing import TYPE_CHECKING, Self

from joblib._multiprocessing_helpers import mp

from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import DatasetMissesDataError, LearningError, ModelNotFittedError
from safeds.ml.classical import SupervisedModel
from safeds.ml.metrics import ClassificationMetrics, ClassifierMetric

if TYPE_CHECKING:
    from typing import Any

    from safeds.data.tabular.containers import Table


class Classifier(SupervisedModel, ABC):
    """A model for classification tasks."""

    # ------------------------------------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_metrics(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> Table:
        """
        Summarize the classifier's metrics on the given data.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        metrics:
            A table containing the classifier's metrics.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.summarize(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    def accuracy(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the accuracy of the classifier on the given data.

        The accuracy is the proportion of predicted target values that were correct. The **higher** the accuracy, the
        better. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        accuracy:
            The classifier's accuracy.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.accuracy(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def f1_score(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> float:
        """
        Compute the classifier's F₁ score on the given data.

        The F₁ score is the harmonic mean of precision and recall. The **higher** the F₁ score, the better the
        classifier. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        f1_score:
            The classifier's F₁ score.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.f1_score(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    def precision(
        self,
        validation_or_test_set: Table | TabularDataset,
        positive_class: Any,
    ) -> float:
        """
        Compute the classifier's precision on the given data.

        The precision is the proportion of positive predictions that were correct. The **higher** the precision, the
        better the classifier. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        precision:
            The classifier's precision.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.precision(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    def recall(self, validation_or_test_set: Table | TabularDataset, positive_class: Any) -> float:
        """
        Compute the classifier's recall on the given data.

        The recall is the proportion of actual positives that were predicted correctly. The **higher** the recall, the
        better the classifier. Results range from 0.0 to 1.0.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.
        positive_class:
            The class to be considered positive. All other classes are considered negative.

        Returns
        -------
        recall:
            The classifier's recall.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return ClassificationMetrics.recall(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
            positive_class,
        )

    def fit_by_exhaustive_search(
        self,
        training_set: TabularDataset,
        optimization_metric: ClassifierMetric,
        positive_class: Any = None,
    ) -> Self:
        """
        Use the hyperparameter choices to create multiple models and fit them.

        **Note:** This model is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the features and target.
        optimization_metric:
            The metric that should be used for determining the performance of a model.
        positive_class:
            The class to be considered positive. All other classes are considered negative.
            Needs to be provided when choosing precision, f1score or recall as optimization metric.

        Returns
        -------
        best_model:
            The model that performed the best out of all possible models given the Choices of hyperparameters.

        Raises
        ------
        PlainTableError
            If a table is passed instead of a TabularDataset.
        DatasetMissesDataError
            If the given training set contains no data.
        FittingWithoutChoiceError
            When trying to call this method on a model without hyperparameter choices.
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        if training_set.to_table().row_count == 0:
            raise DatasetMissesDataError
        if optimization_metric.value in {"precision", "recall", "f1score"} and positive_class is None:
            raise LearningError(
                f"Please provide a positive class when using optimization metric '{optimization_metric.value}'",
            )

        self._check_additional_fit_by_exhaustive_search_preconditions()

        [train_split, test_split] = training_set.to_table().split_rows(0.75)
        train_data = train_split.to_tabular_dataset(
            target_name=training_set.target.name,
            extra_names=training_set.extras.column_names,
        )
        test_data = test_split.to_tabular_dataset(
            target_name=training_set.target.name,
            extra_names=training_set.extras.column_names,
        )

        list_of_models = self._get_models_for_all_choices()
        list_of_fitted_models = []

        with ProcessPoolExecutor(max_workers=len(list_of_models), mp_context=mp.get_context("spawn")) as executor:
            futures = []
            for model in list_of_models:
                futures.append(executor.submit(model.fit, train_data))
            [done, _] = wait(futures, return_when=ALL_COMPLETED)
            for future in done:
                list_of_fitted_models.append(future.result())
        executor.shutdown()
        best_model = None
        best_metric_value = None
        for fitted_model in list_of_fitted_models:
            if best_model is None:
                best_model = fitted_model
                match optimization_metric.value:
                    case "accuracy":
                        best_metric_value = fitted_model.accuracy(test_data)
                    case "precision":
                        best_metric_value = fitted_model.precision(test_data, positive_class)
                    case "recall":
                        best_metric_value = fitted_model.recall(test_data, positive_class)
                    case "f1_score":
                        best_metric_value = fitted_model.recall(test_data, positive_class)
            else:
                match optimization_metric.value:
                    case "accuracy":
                        accuracy_of_fitted_model = fitted_model.accuracy(test_data)
                        if accuracy_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = accuracy_of_fitted_model  # pragma: no cover
                    case "precision":
                        precision_of_fitted_model = fitted_model.precision(test_data, positive_class)
                        if precision_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = precision_of_fitted_model  # pragma: no cover
                    case "recall":
                        recall_of_fitted_model = fitted_model.recall(test_data, positive_class)
                        if recall_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = recall_of_fitted_model  # pragma: no cover
                    case "f1_score":
                        f1score_of_fitted_model = fitted_model.f1_score(test_data, positive_class)
                        if f1score_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = f1score_of_fitted_model  # pragma: no cover
        assert best_model is not None
        return best_model


def _extract_table(table_or_dataset: Table | TabularDataset) -> Table:
    """Extract the table from the given table or dataset."""
    if isinstance(table_or_dataset, TabularDataset):
        return table_or_dataset.to_table()
    else:
        return table_or_dataset

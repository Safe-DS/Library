from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError

from ._classifier import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin
    from sklearn.ensemble import AdaBoostClassifier as sk_AdaBoostClassifier

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers import Table


class AdaBoostClassifier(Classifier):
    """
    Ada Boost classification.

    Parameters
    ----------
    learner:
        The learner from which the boosted ensemble is built.
    maximum_number_of_learners:
        The maximum number of learners at which boosting is terminated. In case of perfect fit, the learning procedure
        is stopped early. Has to be greater than 0.
    learning_rate:
        Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution
        of each classifier. Has to be greater than 0.

    Raises
    ------
    OutOfBoundsError
        If `maximum_number_of_learners` or `learning_rate` are less than or equal to 0.
    """

    def __init__(
        self,
        *,
        learner: Classifier | None = None,
        maximum_number_of_learners: int = 50,
        learning_rate: float = 1.0,
    ) -> None:
        super().__init__()

        # Validation
        if maximum_number_of_learners < 1:
            raise OutOfBoundsError(
                maximum_number_of_learners,
                name="maximum_number_of_learners",
                lower_bound=ClosedBound(1),
            )
        if learning_rate <= 0:
            raise OutOfBoundsError(learning_rate, name="learning_rate", lower_bound=OpenBound(0))

        # Hyperparameters
        self._learner = learner
        self._maximum_number_of_learners = maximum_number_of_learners
        self._learning_rate = learning_rate

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._learning_rate,
            self._maximum_number_of_learners,
        )

    @property
    def learner(self) -> Classifier | None:
        """
        Get the base learner used for training the ensemble.

        Returns
        -------
        result:
            The base learner.
        """
        return self._learner

    @property
    def maximum_number_of_learners(self) -> int:
        """
        Get the maximum number of learners in the ensemble.

        Returns
        -------
        result:
            The maximum number of learners.
        """
        return self._maximum_number_of_learners

    @property
    def learning_rate(self) -> float:
        """
        Get the learning rate.

        Returns
        -------
        result:
            The learning rate.
        """
        return self._learning_rate

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    # def fit(self, training_set: TabularDataset) -> AdaBoostClassifier:
    #     """
    #     Create a copy of this classifier and fit it with the given training data.
    #
    #     This classifier is not modified.
    #
    #     Parameters
    #     ----------
    #     training_set:
    #         The training data containing the feature and target vectors.
    #
    #     Returns
    #     -------
    #     fitted_classifier:
    #         The fitted classifier.
    #
    #     Raises
    #     ------
    #     LearningError
    #         If the training data contains invalid values or if the training failed.
    #     TypeError
    #         If a table is passed instead of a tabular dataset.
    #     NonNumericColumnError
    #         If the training data contains non-numerical values.
    #     MissingValuesColumnError
    #         If the training data contains missing values.
    #     DatasetMissesDataError
    #         If the training data contains no rows.
    #     """
    #     wrapped_classifier = self._get_sklearn_model()
    #     fit(wrapped_classifier, training_set)
    #
    #     result = AdaBoostClassifier(
    #         learner=self.learner,
    #         maximum_number_of_learners=self.maximum_number_of_learners,
    #         learning_rate=self._learning_rate,
    #     )
    #     result._wrapped_classifier = wrapped_classifier
    #     result._feature_schema = training_set.features.column_names
    #     result._target_name = training_set.target.name
    #
    #     return result

    def _get_sklearn_model(self) -> ClassifierMixin:
        from sklearn.ensemble import AdaBoostClassifier as sk_AdaBoostClassifier

        learner = self.learner._get_sklearn_model() if self.learner is not None else None
        return sk_AdaBoostClassifier(
            estimator=learner,
            n_estimators=self.maximum_number_of_learners,
            learning_rate=self._learning_rate,
        )

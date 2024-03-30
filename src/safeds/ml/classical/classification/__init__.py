"""Classes for classification tasks."""

from ._ada_boost import AdaBoostClassifier
from ._classifier import Classifier
from ._decision_tree import DecisionTreeClassifier
from ._gradient_boosting import GradientBoostingClassifier
from ._k_nearest_neighbors import KNearestNeighborsClassifier
from ._logistic_regression import LogisticRegressionClassifier
from ._random_forest import RandomForestClassifier
from ._support_vector_machine import SupportVectorMachineClassifier

__all__ = [
    "AdaBoostClassifier",
    "Classifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "KNearestNeighborsClassifier",
    "LogisticRegressionClassifier",
    "RandomForestClassifier",
    "SupportVectorMachineClassifier",
]

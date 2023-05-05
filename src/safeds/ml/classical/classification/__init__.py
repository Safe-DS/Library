"""Classes for classification tasks."""

from ._ada_boost import AdaBoost
from ._classifier import Classifier
from ._decision_tree import DecisionTree
from ._gradient_boosting import GradientBoosting
from ._k_nearest_neighbors import KNearestNeighbors
from ._logistic_regression import LogisticRegression
from ._random_forest import RandomForest
from ._support_vector_machine import SupportVectorMachine

__all__ = [
    "AdaBoost",
    "Classifier",
    "DecisionTree",
    "GradientBoosting",
    "KNearestNeighbors",
    "LogisticRegression",
    "RandomForest",
    "SupportVectorMachine",
]

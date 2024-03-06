"""Models for regression tasks."""

from ._ada_boost import AdaBoost
from ._decision_tree import DecisionTree
from ._elastic_net_regression import ElasticNetRegression
from ._gradient_boosting import GradientBoosting
from ._k_nearest_neighbors import KNearestNeighbors
from ._lasso_regression import LassoRegression
from ._linear_regression import LinearRegression
from ._random_forest import RandomForest
from ._regressor import Regressor
from ._ridge_regression import RidgeRegression
from ._support_vector_machine import SupportVectorMachine
from ._arima import arimaModel

__all__ = [
    "AdaBoost",
    "arimaModel",
    "DecisionTree",
    "ElasticNetRegression",
    "GradientBoosting",
    "KNearestNeighbors",
    "LassoRegression",
    "LinearRegression",
    "RandomForest",
    "Regressor",
    "RidgeRegression",
    "SupportVectorMachine",
]

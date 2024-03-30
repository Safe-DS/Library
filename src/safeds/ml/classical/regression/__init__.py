"""Models for regression tasks."""

from ._ada_boost import AdaBoostRegressor
from ._decision_tree import DecisionTreeRegressor
from ._elastic_net_regression import ElasticNetRegressor
from ._gradient_boosting import GradientBoostingRegressor
from ._k_nearest_neighbors import KNearestNeighborsRegressor
from ._lasso_regression import LassoRegressor
from ._linear_regression import LinearRegressionRegressor
from ._random_forest import RandomForestRegressor
from ._regressor import Regressor
from ._ridge_regression import RidgeRegressor
from ._support_vector_machine import SupportVectorMachineRegressor

__all__ = [
    "AdaBoostRegressor",
    "DecisionTreeRegressor",
    "ElasticNetRegressor",
    "GradientBoostingRegressor",
    "KNearestNeighborsRegressor",
    "LassoRegressor",
    "LinearRegressionRegressor",
    "RandomForestRegressor",
    "Regressor",
    "RidgeRegressor",
    "SupportVectorMachineRegressor",
]

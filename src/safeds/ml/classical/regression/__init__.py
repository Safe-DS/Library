"""Models for regression tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._ada_boost import AdaBoostRegressor
    from ._arima import ArimaModelRegressor
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

apipkg.initpkg(
    __name__,
    {
        "AdaBoostRegressor": "._ada_boost:AdaBoostRegressor",
        "ArimaModelRegressor": "._arima:ArimaModelRegressor",
        "DecisionTreeRegressor": "._decision_tree:DecisionTreeRegressor",
        "ElasticNetRegressor": "._elastic_net_regression:ElasticNetRegressor",
        "GradientBoostingRegressor": "._gradient_boosting:GradientBoostingRegressor",
        "KNearestNeighborsRegressor": "._k_nearest_neighbors:KNearestNeighborsRegressor",
        "LassoRegressor": "._lasso_regression:LassoRegressor",
        "LinearRegressionRegressor": "._linear_regression:LinearRegressionRegressor",
        "RandomForestRegressor": "._random_forest:RandomForestRegressor",
        "Regressor": "._regressor:Regressor",
        "RidgeRegressor": "._ridge_regression:RidgeRegressor",
        "SupportVectorMachineRegressor": "._support_vector_machine:SupportVectorMachineRegressor",
    },
)

__all__ = [
    "AdaBoostRegressor",
    "ArimaModelRegressor",
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

"""Models for regression tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._ada_boost_regressor import AdaBoostRegressor
    from ._arima import ArimaModelRegressor
    from ._decision_tree_regressor import DecisionTreeRegressor
    from ._elastic_net_regressor import ElasticNetRegressor
    from ._gradient_boosting_regressor import GradientBoostingRegressor
    from ._k_nearest_neighbors_regressor import KNearestNeighborsRegressor
    from ._random_forest_regressor import RandomForestRegressor
    from ._regressor import Regressor
    from ._support_vector_regressor import SupportVectorRegressor

apipkg.initpkg(
    __name__,
    {
        "AdaBoostRegressor": "._ada_boost_regressor:AdaBoostRegressor",
        "ArimaModelRegressor": "._arima:ArimaModelRegressor",
        "DecisionTreeRegressor": "._decision_tree_regressor:DecisionTreeRegressor",
        "ElasticNetRegressor": "._elastic_net_regressor:ElasticNetRegressor",
        "GradientBoostingRegressor": "._gradient_boosting_regressor:GradientBoostingRegressor",
        "KNearestNeighborsRegressor": "._k_nearest_neighbors_regressor:KNearestNeighborsRegressor",
        "RandomForestRegressor": "._random_forest_regressor:RandomForestRegressor",
        "Regressor": "._regressor:Regressor",
        "SupportVectorRegressor": "._support_vector_regressor:SupportVectorRegressor",
    },
)

__all__ = [
    "AdaBoostRegressor",
    "ArimaModelRegressor",
    "DecisionTreeRegressor",
    "ElasticNetRegressor",
    "GradientBoostingRegressor",
    "KNearestNeighborsRegressor",
    "RandomForestRegressor",
    "Regressor",
    "SupportVectorRegressor",
]

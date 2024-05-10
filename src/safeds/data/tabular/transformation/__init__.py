"""Classes for transforming tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._discretizer import Discretizer
    from ._invertible_table_transformer import InvertibleTableTransformer
    from ._label_encoder import LabelEncoder
    from ._one_hot_encoder import OneHotEncoder
    from ._range_scaler import RangeScaler
    from ._simple_imputer import SimpleImputer
    from ._standard_scaler import StandardScaler
    from ._table_transformer import TableTransformer

apipkg.initpkg(
    __name__,
    {
        "Discretizer": "._discretizer:Discretizer",
        "InvertibleTableTransformer": "._table_transformer:InvertibleTableTransformer",
        "LabelEncoder": "._label_encoder:LabelEncoder",
        "OneHotEncoder": "._one_hot_encoder:OneHotEncoder",
        "RangeScaler": "._range_scaler:RangeScaler",
        "SimpleImputer": "._simple_imputer:SimpleImputer",
        "StandardScaler": "._standard_scaler:StandardScaler",
        "TableTransformer": "._table_transformer:TableTransformer",
    },
)

__all__ = [
    "Discretizer",
    "InvertibleTableTransformer",
    "LabelEncoder",
    "OneHotEncoder",
    "RangeScaler",
    "SimpleImputer",
    "StandardScaler",
    "TableTransformer",
]

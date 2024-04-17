"""Classes for transforming tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._discretizer import Discretizer
    from ._imputer import Imputer
    from ._label_encoder import LabelEncoder
    from ._one_hot_encoder import OneHotEncoder
    from ._range_scaler import RangeScaler
    from ._standard_scaler import StandardScaler
    from ._table_transformer import InvertibleTableTransformer, TableTransformer

apipkg.initpkg(
    __name__,
    {
        "Discretizer": "._discretizer:Discretizer",
        "Imputer": "._imputer:Imputer",
        "InvertibleTableTransformer": "._table_transformer:InvertibleTableTransformer",
        "LabelEncoder": "._label_encoder:LabelEncoder",
        "OneHotEncoder": "._one_hot_encoder:OneHotEncoder",
        "RangeScaler": "._range_scaler:RangeScaler",
        "StandardScaler": "._standard_scaler:StandardScaler",
        "TableTransformer": "._table_transformer:TableTransformer",
    },
)

__all__ = [
    "Discretizer",
    "Imputer",
    "InvertibleTableTransformer",
    "LabelEncoder",
    "OneHotEncoder",
    "RangeScaler",
    "StandardScaler",
    "TableTransformer",
]

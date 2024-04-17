"""Tools to work with hyperparameters of ML models."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._choice import Choice

apipkg.initpkg(
    __name__,
    {
        "Choice": "._choice:Choice",
    },
)

__all__ = ["Choice"]

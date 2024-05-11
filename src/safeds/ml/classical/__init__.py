"""Classes for classical machine learning, i.e. machine learning that does not use neural networks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._supervised_model import SupervisedModel

apipkg.initpkg(
    __name__,
    {
        "SupervisedModel": "._supervised_model:SupervisedModel",
    },
)

__all__ = [
    "SupervisedModel",
]

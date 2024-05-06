from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from safeds.data.image.typing import ImageSize


class Layer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def input_size(self) -> int | ImageSize:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def output_size(self) -> int | ImageSize:
        pass  # pragma: no cover

    @abstractmethod
    def _set_input_size(self, input_size: int | ImageSize) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def __hash__(self) -> int:
        pass  # pragma: no cover

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def __sizeof__(self) -> int:
        pass  # pragma: no cover

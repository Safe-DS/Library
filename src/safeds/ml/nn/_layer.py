from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from safeds.data.image.typing import ImageSize


class _Layer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def _get_internal_layer(self, **kwargs) -> nn.Module:
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

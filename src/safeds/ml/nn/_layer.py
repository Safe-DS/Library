from abc import ABC, abstractmethod

from torch import nn


class Layer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def _get_internal_layer(self, activation_function: str) -> nn.Module:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def input_size(self) -> int:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def output_size(self) -> int:
        pass  # pragma: no cover

    @abstractmethod
    def _set_input_size(self, input_size: int) -> None:
        pass  # pragma: no cover
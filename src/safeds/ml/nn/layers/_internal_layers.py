# These class must not be nested inside a function, since pickle cannot serialize local classes. Because of this, the
# slow import of torch must be on the global level. To still evaluate the torch import lazily, the classes are moved to
# a separate file.

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn  # slow import

from safeds._config import _init_default_device


class _InternalConvolutional2DLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        activation_function: Literal["sigmoid", "relu", "softmax"],
        padding: int,
        stride: int,
        transpose: bool,
        output_padding: int,
    ):
        super().__init__()

        _init_default_device()

        if transpose:
            self._layer = nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                output_padding=output_padding,
            )
        else:
            self._layer = nn.Conv2d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
        match activation_function:
            case "sigmoid":
                self._fn = nn.Sigmoid()
            case "relu":
                self._fn = nn.ReLU()
            case "softmax":
                self._fn = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        return self._fn(self._layer(x))


class _InternalDropoutLayer(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()

        _init_default_device()

        self._layer = nn.Dropout(probability)

    def forward(self, x: Tensor) -> Tensor:
        return self._layer(x)


class _InternalFlattenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        _init_default_device()

        self._layer = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        return self._layer(x)


class _InternalForwardLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation_function: str):
        super().__init__()

        _init_default_device()

        self._layer = nn.Linear(input_size, output_size)
        match activation_function:
            case "sigmoid":
                self._fn = nn.Sigmoid()
            case "relu":
                self._fn = nn.ReLU()
            case "softmax":
                self._fn = nn.Softmax()
            case "none":
                self._fn = None
            case _:
                raise ValueError("Unknown Activation Function: " + activation_function)

    def forward(self, x: Tensor) -> Tensor:
        return self._fn(self._layer(x)) if self._fn is not None else self._layer(x)


class _InternalLSTMLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation_function: str):
        super().__init__()

        _init_default_device()

        self._layer = nn.LSTM(input_size, output_size)
        match activation_function:
            case "sigmoid":
                self._fn = nn.Sigmoid()
            case "relu":
                self._fn = nn.ReLU()
            case "softmax":
                self._fn = nn.Softmax()
            case "none":
                self._fn = None
            case _:
                raise ValueError("Unknown Activation Function: " + activation_function)

    def forward(self, x: Tensor) -> Tensor:
        return self._fn(self._layer(x)[0]) if self._fn is not None else self._layer(x)[0]


class _InternalPooling2DLayer(nn.Module):
    def __init__(self, strategy: Literal["max", "avg"], kernel_size: int, padding: int, stride: int):
        super().__init__()

        _init_default_device()

        match strategy:
            case "max":
                self._layer = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride)
            case "avg":
                self._layer = nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self._layer(x)


class _InternalGRULayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation_function: str):
        super().__init__()

        _init_default_device()

        self._layer = nn.GRU(input_size, output_size)
        match activation_function:
            case "sigmoid":
                self._fn = nn.Sigmoid()
            case "relu":
                self._fn = nn.ReLU()
            case "softmax":
                self._fn = nn.Softmax()
            case "none":
                self._fn = None
            case _:
                raise ValueError("Unknown Activation Function: " + activation_function)

    def forward(self, x: Tensor) -> Tensor:
        return self._fn(self._layer(x)[0]) if self._fn is not None else self._layer(x)[0]

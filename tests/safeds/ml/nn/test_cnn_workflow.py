import re
from typing import TYPE_CHECKING

import pytest
import torch
from safeds._config import _get_device
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.ml.nn import (
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
)
from safeds.ml.nn.converters import (
    InputConversionImageToColumn,
    InputConversionImageToImage,
    InputConversionImageToTable,
)
from safeds.ml.nn.layers import (
    AveragePooling2DLayer,
    Convolutional2DLayer,
    ConvolutionalTranspose2DLayer,
    FlattenLayer,
    ForwardLayer,
    MaxPooling2DLayer,
)
from safeds.ml.nn.typing import VariableImageSize
from torch.types import Device

from tests.helpers import configure_test_with_device, device_cpu, device_cuda, images_all, resolve_resource_path

if TYPE_CHECKING:
    from safeds.ml.nn.layers import Layer


class TestImageToTableClassifier:
    @pytest.mark.parametrize(
        ("seed", "device", "prediction_label"),
        [
            (
                1234,
                device_cuda,
                ["grayscale"] * 7,
            ),
            (
                4711,
                device_cuda,
                ["white_square"] * 7,
            ),
            (
                1234,
                device_cpu,
                ["grayscale"] * 7,
            ),
            (
                4711,
                device_cpu,
                ["white_square"] * 7,
            ),
        ],
        ids=["seed-1234-cuda", "seed-4711-cuda", "seed-1234-cpu", "seed-4711-cpu"],
    )
    def test_should_train_and_predict_model(
        self,
        seed: int,
        prediction_label: list[str],
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        torch.manual_seed(seed)

        image_list, filenames = ImageList.from_files(resolve_resource_path(images_all()), return_filenames=True)
        image_list = image_list.resize(20, 20)
        classes = []
        for filename in filenames:
            groups = re.search(r"(.*)[\\/](.*)\.", filename)
            if groups is not None:
                classes.append(groups.group(2))
        image_classes = Table({"class": classes})
        one_hot_encoder = OneHotEncoder(column_names="class").fit(image_classes)
        image_classes_one_hot_encoded = one_hot_encoder.transform(image_classes)
        image_dataset = ImageDataset(image_list, image_classes_one_hot_encoded)
        num_of_classes: int = image_dataset.output_size if isinstance(image_dataset.output_size, int) else 0
        layers = [Convolutional2DLayer(1, 2), MaxPooling2DLayer(10), FlattenLayer(), ForwardLayer(num_of_classes)]
        nn_original = NeuralNetworkClassifier(
            InputConversionImageToTable(image_dataset.input_size),
            layers,
        )
        nn = nn_original.fit(image_dataset, epoch_size=2)
        assert nn_original._model is not nn._model
        prediction: ImageDataset = nn.predict(image_dataset.get_input())
        assert one_hot_encoder.inverse_transform(prediction.get_output()) == Table({"class": prediction_label})
        assert prediction._output._tensor.device == _get_device()


class TestImageToColumnClassifier:
    @pytest.mark.parametrize(
        ("seed", "device", "prediction_label"),
        [
            (
                1234,
                device_cuda,
                ["grayscale"] * 7,
            ),
            (
                4711,
                device_cuda,
                ["white_square"] * 7,
            ),
            (
                1234,
                device_cpu,
                ["grayscale"] * 7,
            ),
            (
                4711,
                device_cpu,
                ["white_square"] * 7,
            ),
        ],
        ids=["seed-1234-cuda", "seed-4711-cuda", "seed-1234-cpu", "seed-4711-cpu"],
    )
    def test_should_train_and_predict_model(
        self,
        seed: int,
        prediction_label: list[str],
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        torch.manual_seed(seed)

        image_list, filenames = ImageList.from_files(resolve_resource_path(images_all()), return_filenames=True)
        image_list = image_list.resize(20, 20)
        classes = []
        for filename in filenames:
            groups = re.search(r"(.*)[\\/](.*)\.", filename)
            if groups is not None:
                classes.append(groups.group(2))
        image_classes = Column("class", classes)
        image_dataset = ImageDataset(image_list, image_classes, shuffle=True)
        num_of_classes: int = image_dataset.output_size if isinstance(image_dataset.output_size, int) else 0

        layers = [Convolutional2DLayer(1, 2), AveragePooling2DLayer(10), FlattenLayer(), ForwardLayer(num_of_classes)]
        nn_original = NeuralNetworkClassifier(
            InputConversionImageToColumn(image_dataset.input_size),
            layers,
        )
        nn = nn_original.fit(image_dataset, epoch_size=2)
        assert nn_original._model is not nn._model
        prediction: ImageDataset = nn.predict(image_dataset.get_input())
        assert prediction.get_output() == Column("class", prediction_label)
        assert prediction._output._tensor.device == _get_device()


class TestImageToImageRegressor:
    @pytest.mark.parametrize(
        ("seed", "device"),
        [
            (1234, device_cuda),
            (4711, device_cuda),
            (1234, device_cpu),
            (4711, device_cpu),
        ],
        ids=["seed-1234-cuda", "seed-4711-cuda", "seed-1234-cpu", "seed-4711-cpu"],
    )
    def test_should_train_and_predict_model(
        self,
        seed: int,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        torch.manual_seed(seed)

        image_list = ImageList.from_files(resolve_resource_path(images_all()))
        image_list = image_list.resize(20, 20)
        image_list_grayscale = image_list.convert_to_grayscale()
        image_dataset = ImageDataset(image_list, image_list_grayscale)

        layers: list[Layer] = [
            Convolutional2DLayer(6, 2),
            Convolutional2DLayer(12, 2),
            ConvolutionalTranspose2DLayer(6, 2),
            ConvolutionalTranspose2DLayer(4, 2),
        ]
        nn_original = NeuralNetworkRegressor(
            InputConversionImageToImage(image_dataset.input_size),
            layers,
        )
        nn = nn_original.fit(image_dataset, epoch_size=20)
        assert nn_original._model is not nn._model
        prediction = nn.predict(image_dataset.get_input())
        assert isinstance(prediction.get_output(), ImageList)
        assert prediction._output._tensor.device == _get_device()

    @pytest.mark.parametrize(
        ("seed", "device"),
        [
            (4711, device_cuda),
            (4711, device_cpu),
        ],
        ids=["seed-4711-cuda", "seed-4711-cpu"],
    )
    @pytest.mark.parametrize("multi_width", [1, 2, 3])
    @pytest.mark.parametrize("multi_height", [1, 2, 3])
    def test_should_train_and_predict_model_variable_image_size(
        self,
        seed: int,
        device: Device,
        multi_width: int,
        multi_height: int,
    ) -> None:
        configure_test_with_device(device)
        torch.manual_seed(seed)

        image_list = ImageList.from_files(resolve_resource_path(images_all()))
        image_list = image_list.resize(20, 20)
        image_list_grayscale = image_list.convert_to_grayscale()
        image_dataset = ImageDataset(image_list, image_list_grayscale)

        layers: list[Layer] = [
            Convolutional2DLayer(6, 2),
            Convolutional2DLayer(12, 2),
            ConvolutionalTranspose2DLayer(6, 2),
            ConvolutionalTranspose2DLayer(4, 2),
        ]
        nn_original = NeuralNetworkRegressor(
            InputConversionImageToImage(VariableImageSize.from_image_size(image_dataset.input_size)),
            layers,
        )
        nn = nn_original.fit(image_dataset, epoch_size=20)
        assert nn_original._model is not nn._model
        prediction = nn.predict(
            image_dataset.get_input().resize(
                image_dataset.input_size.width * multi_width,
                image_dataset.input_size.height * multi_height,
            ),
        )
        pred_output = prediction.get_output()
        assert isinstance(pred_output, ImageList)
        if isinstance(pred_output, _SingleSizeImageList):
            assert pred_output.widths[0] == image_dataset.input_size.width * multi_width
            assert pred_output.heights[0] == image_dataset.input_size.height * multi_height
        assert prediction.input_size.height == image_dataset.input_size.height * multi_height
        assert prediction._output._tensor.device == _get_device()

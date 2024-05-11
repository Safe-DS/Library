import re
from typing import TYPE_CHECKING

import pytest
import torch
from safeds._config import _get_device
from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.ml.nn import (
    AvgPooling2DLayer,
    Convolutional2DLayer,
    ConvolutionalTranspose2DLayer,
    FlattenLayer,
    ForwardLayer,
    InputConversionImage,
    MaxPooling2DLayer,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    OutputConversionImageToTable,
)
from safeds.ml.nn._output_conversion_image import OutputConversionImageToColumn, OutputConversionImageToImage
from syrupy import SnapshotAssertion
from torch.types import Device

from tests.helpers import configure_test_with_device, device_cpu, device_cuda, images_all, resolve_resource_path

if TYPE_CHECKING:
    from safeds.ml.nn import Layer


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
        one_hot_encoder = OneHotEncoder().fit(image_classes, ["class"])
        image_classes_one_hot_encoded = one_hot_encoder.transform(image_classes)
        image_dataset = ImageDataset(image_list, image_classes_one_hot_encoded)
        num_of_classes: int = image_dataset.output_size if isinstance(image_dataset.output_size, int) else 0
        layers = [Convolutional2DLayer(1, 2), MaxPooling2DLayer(10), FlattenLayer(), ForwardLayer(num_of_classes)]
        nn_original = NeuralNetworkClassifier(
            InputConversionImage(image_dataset.input_size),
            layers,
            OutputConversionImageToTable(),
        )
        nn = nn_original.fit(image_dataset, epoch_size=2)
        assert str(nn_original._model.state_dict().values()) != str(nn._model.state_dict().values())
        assert not torch.all(
            torch.eq(
                nn_original._model.state_dict()["_pytorch_layers.3._layer.bias"],
                nn._model.state_dict()["_pytorch_layers.3._layer.bias"],
            ),
        ).item()
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

        layers = [Convolutional2DLayer(1, 2), AvgPooling2DLayer(10), FlattenLayer(), ForwardLayer(num_of_classes)]
        nn_original = NeuralNetworkClassifier(
            InputConversionImage(image_dataset.input_size),
            layers,
            OutputConversionImageToColumn(),
        )
        nn = nn_original.fit(image_dataset, epoch_size=2)
        assert str(nn_original._model.state_dict().values()) != str(nn._model.state_dict().values())
        assert not torch.all(
            torch.eq(
                nn_original._model.state_dict()["_pytorch_layers.3._layer.bias"],
                nn._model.state_dict()["_pytorch_layers.3._layer.bias"],
            ),
        ).item()
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
        snapshot_png_image_list: SnapshotAssertion,
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
            InputConversionImage(image_dataset.input_size),
            layers,
            OutputConversionImageToImage(),
        )
        nn = nn_original.fit(image_dataset, epoch_size=20)
        assert str(nn_original._model.state_dict().values()) != str(nn._model.state_dict().values())
        assert not torch.all(
            torch.eq(
                nn_original._model.state_dict()["_pytorch_layers.3._layer.bias"],
                nn._model.state_dict()["_pytorch_layers.3._layer.bias"],
            ),
        ).item()
        prediction = nn.predict(image_dataset.get_input())
        assert isinstance(prediction.get_output(), ImageList)
        assert prediction._output._tensor.device == _get_device()

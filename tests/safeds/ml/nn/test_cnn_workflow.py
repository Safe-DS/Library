import re

import pytest
import torch
from syrupy import SnapshotAssertion
from torch.types import Device

from safeds.data.image.containers import ImageList, ImageDataset
from safeds.data.tabular.containers import Table, Column
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.ml.nn import NeuralNetworkClassifier, InputConversionImage, Convolutional2DLayer, MaxPooling2DLayer, \
    FlattenLayer, ForwardLayer, OutputConversionImageToTable, ConvolutionalTranspose2DLayer, NeuralNetworkRegressor
from safeds.ml.nn._output_conversion_image import OutputConversionImageToColumn, OutputConversionImageToImage
from tests.helpers import resolve_resource_path, images_all, device_cuda, device_cpu, skip_if_device_not_available


class TestImageToTable:

    @pytest.mark.parametrize(
        ("seed", "device", "layer_3_bias", "prediction_label"),
        [
            (1234, device_cuda, [0.5809096097946167, -0.32418742775917053, 0.026058292016386986, 0.5801554918289185], ["grayscale"] * 7),
            (4711, device_cuda, [-0.8114155530929565, -0.9443624019622803, 0.8557258248329163, -0.848240852355957], ["white_square"] * 7),
            (1234, device_cpu, [-0.6926110982894897, 0.33004942536354065, -0.32962560653686523, 0.5768553614616394], ["grayscale"] * 7),
            (4711, device_cpu, [-0.9051575660705566, -0.8625037670135498, 0.24682046473026276, -0.2612163722515106], ["white_square"] * 7),
        ],
        ids=["seed-1234-cuda", "seed-4711-cuda", "seed-1234-cpu", "seed-4711-cpu"]
    )
    def test_should_train_and_predict_model(self, seed: int, layer_3_bias: list[float], prediction_label: list[str], device: Device):
        skip_if_device_not_available(device)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        image_list, filenames = ImageList.from_files(resolve_resource_path(images_all()), return_filenames=True)
        image_list = image_list.resize(20, 20)
        image_classes = Table({"class": [re.search(r"(.*)[\\/](.*)\.", filename).group(2) for filename in filenames]})
        one_hot_encoder = OneHotEncoder().fit(image_classes, ["class"])
        image_classes_one_hot_encoded = one_hot_encoder.transform(image_classes)
        image_dataset = ImageDataset(image_list, image_classes_one_hot_encoded)
        layers = [
            Convolutional2DLayer(1, 2),
            MaxPooling2DLayer(10),
            FlattenLayer(),
            ForwardLayer(image_dataset.output_size)
        ]
        nn_original = NeuralNetworkClassifier(InputConversionImage(image_dataset.input_size), layers,
                                              OutputConversionImageToTable())
        nn = nn_original.fit(image_dataset, epoch_size=2)
        assert str(nn_original._model.state_dict().values()) != str(nn._model.state_dict().values())
        assert nn._model.state_dict()["_pytorch_layers.3._layer.bias"].tolist() == layer_3_bias
        prediction = nn.predict(image_dataset.get_input())
        assert one_hot_encoder.inverse_transform(prediction.get_output()) == Table({"class": prediction_label})


class TestImageToColumn:

    @pytest.mark.parametrize(
        ("seed", "device", "layer_3_bias", "prediction_label"),
        [
            (1234, device_cuda, [0.5809096097946167, -0.32418742775917053, 0.026058292016386986, 0.5801554918289185], ["grayscale"] * 7),
            (4711, device_cuda, [-0.8114155530929565, -0.9443624019622803, 0.8557258248329163, -0.848240852355957], ["white_square"] * 7),
            (1234, device_cpu, [-0.6926110982894897, 0.33004942536354065, -0.32962560653686523, 0.5768553614616394], ["grayscale"] * 7),
            (4711, device_cpu, [-0.9051575660705566, -0.8625037670135498, 0.24682046473026276, -0.2612163722515106], ["white_square"] * 7),
        ],
        ids=["seed-1234-cuda", "seed-4711-cuda", "seed-1234-cpu", "seed-4711-cpu"]
    )
    def test_should_train_and_predict_model(self, seed: int, layer_3_bias: list[float], prediction_label: list[str], device: Device):
        skip_if_device_not_available(device)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        image_list, filenames = ImageList.from_files(resolve_resource_path(images_all()), return_filenames=True)
        image_list = image_list.resize(20, 20)
        image_classes = Column("class", [re.search(r"(.*)[\\/](.*)\.", filename).group(2) for filename in filenames])
        image_dataset = ImageDataset(image_list, image_classes)
        print(image_dataset._output._tensor)
        print(image_dataset._output._tensor.size())

        layers = [
            Convolutional2DLayer(1, 2),
            MaxPooling2DLayer(10),
            FlattenLayer(),
            ForwardLayer(image_dataset.output_size)
        ]
        nn_original = NeuralNetworkClassifier(InputConversionImage(image_dataset.input_size), layers,
                                              OutputConversionImageToColumn())
        nn = nn_original.fit(image_dataset, epoch_size=2)
        assert str(nn_original._model.state_dict().values()) != str(nn._model.state_dict().values())
        assert nn._model.state_dict()["_pytorch_layers.3._layer.bias"].tolist() == layer_3_bias
        prediction = nn.predict(image_dataset.get_input())
        assert prediction.get_output() == Column("class", prediction_label)


class TestImageToImage:

    @pytest.mark.parametrize(
        ("seed", "device", "layer_3_bias"),
        [
            (1234, device_cuda, [0.13570494949817657, 0.02420804090797901, -0.1311846673488617, 0.22676928341388702]),
            (4711, device_cuda, [0.11234158277511597, 0.13972002267837524, -0.07925988733768463, 0.07342307269573212]),
            (1234, device_cpu, [-0.1637762188911438, 0.02012808807194233, -0.22295698523521423, 0.1689515858888626]),
            (4711, device_cpu, [-0.030541712418198586, -0.15364733338356018, 0.1741572618484497, 0.015837203711271286]),
        ],
        ids=["seed-1234-cuda", "seed-4711-cuda", "seed-1234-cpu", "seed-4711-cpu"]
    )
    def test_should_train_and_predict_model(self, seed: int, snapshot_png_image_list: SnapshotAssertion, layer_3_bias: list[float], device: Device):
        skip_if_device_not_available(device)
        torch.set_default_device(device)
        torch.manual_seed(seed)

        image_list = ImageList.from_files(resolve_resource_path(images_all()))
        image_list = image_list.resize(20, 20)
        image_list_grayscale = image_list.convert_to_grayscale()
        image_dataset = ImageDataset(image_list, image_list_grayscale)

        layers = [
            Convolutional2DLayer(6, 2),
            Convolutional2DLayer(12, 2),
            ConvolutionalTranspose2DLayer(6, 2),
            ConvolutionalTranspose2DLayer(4, 2),
        ]
        nn_original = NeuralNetworkRegressor(InputConversionImage(image_dataset.input_size), layers,
                                             OutputConversionImageToImage())
        nn = nn_original.fit(image_dataset, epoch_size=20)
        assert str(nn_original._model.state_dict().values()) != str(nn._model.state_dict().values())
        assert nn._model.state_dict()["_pytorch_layers.3._layer.bias"].tolist() == layer_3_bias
        prediction = nn.predict(image_dataset.get_input())
        assert prediction.get_output() == snapshot_png_image_list

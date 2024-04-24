import re

import pytest
import torch

from safeds._config import _get_device
from safeds.data.image.containers import ImageList, ImageDataset
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.ml.nn import NeuralNetworkClassifier, InputConversionImage, Convolutional2DLayer, MaxPooling2DLayer, \
    FlattenLayer, ForwardLayer, OutputConversionImage
from tests.helpers import resolve_resource_path, images_all


class TestImageToTable:

    @pytest.mark.parametrize(
        ("seed", "layer_3_bias"),
        [
            (1234, [0.5809096097946167, -0.32418742775917053, 0.026058292016386986, 0.5801554918289185]),
            (4711, [-0.8114155530929565, -0.9443624019622803, 0.8557258248329163, -0.848240852355957]),
        ],
        ids=["seed-1234", "seed-4711"]
    )
    def test_should_train_model(self, seed: int, layer_3_bias: list[float]):
        torch.manual_seed(seed)
        torch.set_default_device(_get_device())

        image_list, filenames = ImageList.from_files(resolve_resource_path(images_all()), return_filenames=True)
        image_list = image_list.resize(20, 20)
        image_classes = Table({"class": [re.search(r"(.*)\\(.*)\.", filename).group(2) for filename in filenames]})
        one_hot_encoder = OneHotEncoder().fit(image_classes, ["class"])
        image_classes_one_hot_encoded = one_hot_encoder.transform(image_classes)
        image_dataset = ImageDataset(image_list, image_classes_one_hot_encoded)

        layers = [
            Convolutional2DLayer(1, 2),
            MaxPooling2DLayer(10),
            FlattenLayer(),
            ForwardLayer(len(one_hot_encoder.get_names_of_added_columns()))
        ]
        nn_original = NeuralNetworkClassifier(InputConversionImage(image_dataset.input_size), layers,
                                              OutputConversionImage(False))
        nn = nn_original.fit(image_dataset, epoch_size=2)
        assert str(nn_original._model.state_dict().values()) != str(nn._model.state_dict().values())
        assert nn._model.state_dict()["_pytorch_layers.3._layer.bias"].tolist() == layer_3_bias

from torch import nn
from safeds.data.tabular.containers import TaggedTable
from safeds.ml.nn import fnn_Layer


class fnn_Model():
    def __init__(self, layers: list):
        self._model = PytorchModel(layers)

    def train(self, train_data: TaggedTable, epoch_size=25, batch_size=1):
        # Todo Transform TaggedTable to Dataloader
        # TODO Implement Train Loop
        pass

    def predict(self, test_data: TaggedTable):
        pass


class PytorchModel(nn.Module):
    def __init__(self, layer_list: list[fnn_Layer]):
        super().__init__()
        layers = []
        for layer in layer_list:
            layers.append(layer._get_pytorch_layer())

        self._pytorch_layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.pytorch_layers:
            x = layer(x)
        return x

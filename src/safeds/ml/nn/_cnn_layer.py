import torch.nn as nn


class Layer():
    pass


class Conv2d_Layer(Layer):

    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        self._input_channel, self._output_channel, self._kernel_size, self._stride, self._padding = input_channel, output_channel, kernel_size, stride, padding

    def _create_pytorch_layer(self):
        return PT_Conv2d_Layer(self._input_channel, self._output_channel, self._kernel_size, self._stride,
                               self._padding)


class ConvTranspose2d_Layer(Layer):

    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, output_padding):
        self._input_channel, self._output_channel, self._kernel_size, self._stride, self._padding, self._output_padding = input_channel, output_channel, kernel_size, stride, padding, output_padding

    def _create_pytorch_layer(self):
        return PT_ConvTranspose2d_Layer(self._input_channel, self._output_channel, self._kernel_size, self._stride,
                                        self._padding, self._output_padding)


class MaxPool2d_Layer(Layer):

    def __init__(self, kernel_size, stride, padding):
        self._kernel_size, self._stride, self._padding = kernel_size, stride, padding

    def _create_pytorch_layer(self):
        return PT_MaxPool2d_Layer(self._kernel_size, self._stride, self._padding)


class PT_Conv2d_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super().__init__()
        self._input_dim, self._output_dim, self._kernel_size, self._stride, self._padding = input_dim, output_dim, kernel_size, stride, padding
        self.conv2d = nn.Conv2d(in_channels=self._input_dim, out_channels=self._output_dim,
                                kernel_size=self._kernel_size, stride=self._stride, padding=self._padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class PT_ConvTranspose2d_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, output_padding):
        super().__init__()
        self._input_dim, self._output_dim, self._kernel_size, self._stride, self._padding, self._output_padding = input_dim, output_dim, kernel_size, stride, padding, output_padding
        self.convtrans2d = nn.ConvTranspose2d(in_channels=self._input_dim, out_channels=self._output_dim,
                                              kernel_size=self._kernel_size, stride=self._stride, padding=self._padding, output_padding=output_padding)

    def forward(self, x):
        out = self.convtrans2d(x)
        return out


class PT_MaxPool2d_Layer(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self._kernel_size, self._stride, self._padding = kernel_size, stride, padding
        self.maxpool2d = nn.MaxPool2d(kernel_size=self._kernel_size, stride=self._stride, padding=self._padding)

    def forward(self, x):
        out = self.maxpool2d(x)
        return out

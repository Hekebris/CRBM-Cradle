import torch
import torch.nn as nn


class CRBMConvolutionalLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool_size=2
    ):
        super(CRBMConvolutionalLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = self.pool(x)
        # x = torch.bernoulli(x)

        return x

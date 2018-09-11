import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_relu(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):

        super(conv_relu, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x):
        x = self.conv(x)

        x = F.relu(x)

        return x
class MnistModel_v1(nn.Module):

    _channels = [16, 32, 32, 64, 64]
    def __init__(self):
        super(MnistModel_v1, self).__init__()

        self.conv1 = conv_relu(in_channels=1, out_channels=self._channels[0], kernel_size=3, stride=1, padding=1)

        self.conv2 = conv_relu(in_channels=self._channels[0], out_channels=self._channels[1], kernel_size=3, stride=1, padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = conv_relu(in_channels=self._channels[1], out_channels=self._channels[2], kernel_size=3, stride=1, padding=1)

        self.conv4 = conv_relu(in_channels=self._channels[2], out_channels=self._channels[3], kernel_size=3, stride=1,
                               padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv5 = conv_relu(in_channels=self._channels[3], out_channels=self._channels[4], kernel_size=3, stride=1,
                               padding=1)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv6 = conv_relu(in_channels=self._channels[4], out_channels=self._channels[4], kernel_size=3, stride=1, padding = 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(in_features=64, out_features=10, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.maxpool3(x)

        x = self.conv6(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


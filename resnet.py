# network strucuture
import torch.nn as nn


class residualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.downsample:
            y = self.downsample(y)
        x += y
        return x


class Resnet_20_CIFAR10(nn.Module):
    def __init__(self):
        super(Resnet_20_CIFAR10, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, 16, residualBlock, 3)
        self.layer2 = self._make_layer(16, 32, residualBlock, 3)
        self.layer3 = self._make_layer(32, 64, residualBlock, 3)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, in_channel, out_channel, block, block_num):
        downsample = None
        stride = 1
        if in_channel != out_channel:
            stride = 2
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 2, padding=1),
                nn.BatchNorm2d(out_channel))
        layers = []
        for i in range(block_num):
            if not i:
                layers.append(block(in_channel, out_channel, stride, downsample=downsample))
            else:
                layers.append(block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Copyright 2022 Lorna. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision.transforms import functional as F_vision

__all__ = [
    "PathDiscriminator", "CycleNet",
    "path_discriminator", "cyclenet",
]


class PathDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
    ) -> None:
        super(PathDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(channels, int(channels * 2), (4, 4), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 2), int(channels * 4), (4, 4), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 4)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 4), int(channels * 8), (4, 4), (1, 1), (1, 1)),
            nn.InstanceNorm2d(int(channels * 8)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 8), out_channels, (4, 4), (1, 1), (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.main(x)

        return x


class CycleNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
    ) -> None:
        super(CycleNet, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, channels, (7, 7), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Downsampling
            nn.Conv2d(channels, int(channels * 2), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(int(channels * 2), int(channels * 4), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 4), track_running_stats=True),
            nn.ReLU(True),

            # Residual blocks
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),

            # Upsampling
            nn.ConvTranspose2d(int(channels * 4), int(channels * 2), (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(channels * 2), channels, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, (7, 7), (1, 1), (0, 0)),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.main(x)

        return x


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(_ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.res(x)

        x = torch.add(x, identity)

        return x


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def path_discriminator(**kwargs) -> PathDiscriminator:
    model = PathDiscriminator(**kwargs)
    model.apply(_weights_init)

    return model


def cyclenet(**kwargs) -> CycleNet:
    model = CycleNet(**kwargs)
    model.apply(_weights_init)

    return model

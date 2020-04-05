# Copyright 2020 Lorna Authors. All Rights Reserved.
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
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class Generator(nn.Module):
    def __init__(self, nz, nc, res_blocks=9):
        super(Generator, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nz, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        downsampling = []
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            downsampling = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_channels),
                            nn.ReLU(inplace=True)]
            in_channels = out_channels
            out_channels = in_channels * 2
        self.downsample = nn.Sequential(*downsampling)

        # Residual blocks
        res_block = []
        for _ in range(res_blocks):
            res_block += [ResidualBlock(in_channels)]
        self.res = nn.Sequential(*res_block)

        # Upsampling
        upsampling = []
        out_channels = in_channels // 2
        for _ in range(2):
            upsampling += [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1,
                                              output_padding=1),
                           nn.InstanceNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
            in_channels = out_channels
            out_channels = in_channels // 2
        self.upsample = nn.Sequential(*upsampling)

        # Output layer
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, nc, 7),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.downsample(x)
        x = self.res(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)

# Copyright 2019 Lorna Authors. All Rights Reserved.
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


import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()

    conv_block = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_features, in_features, 3),
                  nn.InstanceNorm2d(in_features),
                  nn.ReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_features, in_features, 3),
                  nn.InstanceNorm2d(in_features)]

    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, x):
    return x + self.conv_block(x)


class Generator(nn.Module):
  def __init__(self, input_nc, output_nc, n_residual_blocks=9):
    super(Generator, self).__init__()

    # Initial convolution block
    model = [nn.ReflectionPad2d(3),
             nn.Conv2d(input_nc, 64, 7),
             nn.InstanceNorm2d(64),
             nn.ReLU(inplace=True)]

    # Downsampling
    in_features = 64
    out_features = in_features * 2
    for _ in range(2):
      model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
      in_features = out_features
      out_features = in_features * 2

    # Residual blocks
    for _ in range(n_residual_blocks):
      model += [ResidualBlock(in_features)]

    # Upsampling
    out_features = in_features // 2
    for _ in range(2):
      model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
      in_features = out_features
      out_features = in_features // 2

    # Output layer
    model += [nn.ReflectionPad2d(3),
              nn.Conv2d(64, output_nc, 7),
              nn.Tanh()]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)


class Discriminator(nn.Module):
  def __init__(self, input_nc):
    super(Discriminator, self).__init__()

    # A bunch of convolutions one after another
    model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
             nn.LeakyReLU(0.2, inplace=True)]

    model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
              nn.InstanceNorm2d(128),
              nn.LeakyReLU(0.2, inplace=True)]

    model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
              nn.InstanceNorm2d(256),
              nn.LeakyReLU(0.2, inplace=True)]

    model += [nn.Conv2d(256, 512, 4, padding=1),
              nn.InstanceNorm2d(512),
              nn.LeakyReLU(0.2, inplace=True)]

    # FCN classification layer
    model += [nn.Conv2d(512, 1, 4, padding=1)]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    x = self.model(x)
    # Average pooling and flatten
    return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
